"""
Subscription management via Dodo Payments + Supabase.

- Stores subscription status in Supabase `subscriptions` table
- Dodo webhooks update the table on subscription changes
- Fast lookups from Supabase instead of API calls
"""

import os
import hmac
import hashlib
from datetime import datetime, timedelta

DODO_API_KEY = os.getenv("DODO_PAYMENTS_API_KEY", "")
DODO_ENVIRONMENT = os.getenv("DODO_ENVIRONMENT", "test_mode")
DODO_WEBHOOK_SECRET = os.getenv("DODO_WEBHOOK_SECRET", "")

# In-memory cache as fallback (5 min TTL)
_subscription_cache = {}
_cache_ttl = timedelta(minutes=5)


def _get_dodo_client():
    """Get Dodo Payments client."""
    try:
        from dodopayments import DodoPayments
        return DodoPayments(
            bearer_token=DODO_API_KEY,
            environment=DODO_ENVIRONMENT,
        )
    except ImportError:
        raise ValueError("dodopayments package not installed. Run: pip install dodopayments")


def _get_supabase():
    """Get Supabase client."""
    from db import get_client
    return get_client()


def _get_subscription_from_db(email: str) -> dict | None:
    """Get subscription from Supabase."""
    try:
        client = _get_supabase()
        result = client.table("subscriptions").select("*").eq("email", email.lower()).execute()
        if result.data:
            row = result.data[0]
            expires_at = row.get("expires_at")
            cancelled_at = row.get("cancelled_at")
            
            # Check if subscription is still valid based on expiry date
            is_premium = row.get("is_premium", False)
            if expires_at:
                try:
                    # Parse expiry date and check if still valid
                    exp_str = str(expires_at)[:10]  # Get YYYY-MM-DD
                    from datetime import date
                    exp_date = date.fromisoformat(exp_str)
                    today = date.today()
                    # Premium if not expired yet
                    is_premium = exp_date >= today
                except (ValueError, TypeError):
                    pass
            
            return {
                "is_premium": is_premium,
                "plan": row.get("plan") if is_premium else None,
                "expires_at": expires_at if is_premium else None,
                "subscription_id": row.get("subscription_id"),
                "cancelled": bool(cancelled_at),
            }
        return None
    except Exception as e:
        print(f"[subscription] DB read error: {e}")
        return None


def _save_subscription_to_db(email: str, is_premium: bool, plan: str | None, 
                              subscription_id: str | None, expires_at: str | None,
                              cancelled_at: str | None = None):
    """Save subscription status to Supabase."""
    try:
        client = _get_supabase()
        data = {
            "email": email.lower(),
            "is_premium": is_premium,
            "plan": plan,
            "subscription_id": subscription_id,
            "expires_at": expires_at,
            "updated_at": datetime.now().isoformat(),
        }
        if cancelled_at is not None:
            data["cancelled_at"] = cancelled_at
        client.table("subscriptions").upsert(data, on_conflict="email").execute()
        print(f"[subscription] Saved: {email} is_premium={is_premium} plan={plan} cancelled={cancelled_at is not None}")
    except Exception as e:
        print(f"[subscription] DB write error: {e}")


def check_subscription(customer_email: str) -> dict:
    """
    Check if a customer has an active subscription.
    
    Priority: 1. In-memory cache, 2. Supabase, 3. Dodo API
    
    Returns:
        {
            "is_premium": bool,
            "plan": "monthly" | "yearly" | None,
            "expires_at": ISO date string | None,
            "subscription_id": str | None,
        }
    """
    default = {"is_premium": False, "plan": None, "expires_at": None, "subscription_id": None}
    
    if not customer_email:
        return default
    
    cache_key = customer_email.lower()
    
    # 1. Check in-memory cache
    if cache_key in _subscription_cache:
        cached, cached_at = _subscription_cache[cache_key]
        if datetime.now() - cached_at < _cache_ttl:
            return cached
    
    # 2. Check Supabase
    db_result = _get_subscription_from_db(customer_email)
    if db_result is not None:
        _subscription_cache[cache_key] = (db_result, datetime.now())
        return db_result
    
    # 3. Fall back to Dodo API (first time user or DB miss)
    if not DODO_API_KEY:
        return default
    
    try:
        client = _get_dodo_client()
        
        # First, find customer by email
        customers = client.customers.list(email=customer_email)
        customer_list = list(customers) if customers else []
        
        if not customer_list:
            # No customer found with this email
            _subscription_cache[cache_key] = (default, datetime.now())
            return default
        
        customer_id = customer_list[0].customer_id
        
        # Now get active subscriptions for this customer
        subscriptions = client.subscriptions.list(
            customer_id=customer_id,
            status="active",
        )
        
        items = list(subscriptions) if subscriptions else []
        
        if not items:
            result = default
        else:
            # Take the first active subscription
            sub = items[0]
            product_id = getattr(sub, "product_id", "") or ""
            
            monthly_id = os.getenv("DODO_MONTHLY_PRODUCT_ID", "")
            yearly_id = os.getenv("DODO_YEARLY_PRODUCT_ID", "")
            
            if product_id == monthly_id:
                plan = "monthly"
            elif product_id == yearly_id:
                plan = "yearly"
            else:
                plan = "premium"
            
            expires = getattr(sub, "current_period_end", None) or getattr(sub, "next_billing_date", None)
            sub_id = getattr(sub, "subscription_id", None) or getattr(sub, "id", None)
            
            result = {
                "is_premium": True,
                "plan": plan,
                "expires_at": str(expires) if expires else None,
                "subscription_id": str(sub_id) if sub_id else None,
            }
            
            # Save to DB for future lookups
            _save_subscription_to_db(
                customer_email, 
                result["is_premium"], 
                result["plan"],
                result["subscription_id"],
                result["expires_at"],
            )
        
        _subscription_cache[cache_key] = (result, datetime.now())
        return result
        
    except Exception as e:
        print(f"[subscription] Error checking subscription: {e}")
        return default


def create_checkout_url(customer_email: str, plan: str = "monthly", customer_name: str = "") -> str | None:
    """
    Create a Dodo checkout URL for subscription.
    
    Args:
        customer_email: Customer's email
        plan: "monthly" or "yearly"
        customer_name: Optional customer name
    
    Returns:
        Checkout URL or None on error
    """
    if not DODO_API_KEY:
        return None
    
    monthly_id = os.getenv("DODO_MONTHLY_PRODUCT_ID", "")
    yearly_id = os.getenv("DODO_YEARLY_PRODUCT_ID", "")
    
    product_id = yearly_id if plan == "yearly" else monthly_id
    
    if not product_id:
        print(f"[subscription] No product ID configured for plan: {plan}")
        return None
    
    try:
        client = _get_dodo_client()
        
        app_url = os.getenv("APP_URL", "http://localhost:5173")
        
        session = client.checkout_sessions.create(
            product_cart=[{"product_id": product_id, "quantity": 1}],
            customer={
                "email": customer_email,
                "name": customer_name or customer_email.split("@")[0],
            },
            return_url=f"{app_url}/pricing?success=true",
            metadata={
                "plan": plan,
                "source": "morrow_desk",
            },
        )
        
        return session.checkout_url
            
    except Exception as e:
        print(f"[subscription] Error creating checkout: {e}")
        return None


def clear_cache(customer_email: str = None):
    """Clear subscription cache, optionally for a specific user."""
    global _subscription_cache
    if customer_email:
        _subscription_cache.pop(customer_email.lower(), None)
    else:
        _subscription_cache = {}


def cancel_subscription(customer_email: str, subscription_id: str = None) -> dict:
    """
    Cancel a user's subscription.
    
    User retains premium access until their current billing period ends.
    
    1. Cancel in Dodo Payments (stops future billing)
    2. Update Supabase to mark as cancelled but keep expires_at
    3. Clear cache
    
    Returns:
        {"success": bool, "error": str | None, "expires_at": str | None}
    """
    if not customer_email:
        return {"success": False, "error": "Email required", "expires_at": None}
    
    # Get current subscription info from DB
    db_result = _get_subscription_from_db(customer_email)
    if not db_result:
        return {"success": False, "error": "No active subscription found", "expires_at": None}
    
    if not subscription_id:
        subscription_id = db_result.get("subscription_id")
    
    expires_at = db_result.get("expires_at")
    plan = db_result.get("plan")
    
    if not subscription_id:
        return {"success": False, "error": "No active subscription found", "expires_at": None}
    
    # 1. Cancel in Dodo Payments (stops future billing)
    if DODO_API_KEY:
        try:
            client = _get_dodo_client()
            # Cancel the subscription - this ends it at current period
            client.subscriptions.update(
                subscription_id=subscription_id,
                status="cancelled",
            )
            print(f"[subscription] Cancelled in Dodo: {subscription_id}")
        except Exception as e:
            print(f"[subscription] Dodo cancel error: {e}")
            # Continue anyway - we'll update DB
    
    # 2. Update Supabase - keep is_premium=True and expires_at, just mark cancelled_at
    # User keeps access until expires_at
    try:
        _save_subscription_to_db(
            email=customer_email,
            is_premium=True,  # Keep premium until expires_at
            plan=plan,
            subscription_id=subscription_id,
            expires_at=expires_at,  # Keep the original expiry date
            cancelled_at=datetime.now().isoformat(),  # Mark when cancelled
        )
        print(f"[subscription] Marked cancelled in DB: {customer_email}, access until {expires_at}")
    except Exception as e:
        print(f"[subscription] DB update error during cancel: {e}")
        return {"success": False, "error": "Database update failed", "expires_at": None}
    
    # 3. Clear cache
    clear_cache(customer_email)
    
    return {"success": True, "error": None, "expires_at": expires_at}


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify Dodo webhook signature."""
    if not DODO_WEBHOOK_SECRET:
        print("[subscription] Warning: No webhook secret configured")
        return True  # Allow in dev mode
    
    try:
        expected = hmac.new(
            DODO_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)
    except Exception as e:
        print(f"[subscription] Signature verification error: {e}")
        return False


def handle_webhook(event_type: str, data: dict) -> bool:
    """
    Handle Dodo webhook event and update subscription status.
    
    Events:
    - subscription.active: User subscribed
    - subscription.cancelled: User cancelled
    - subscription.on_hold: Payment failed
    - subscription.renewed: Subscription renewed
    
    Returns True if handled successfully.
    """
    try:
        # Extract customer email from webhook data
        customer = data.get("customer", {})
        email = customer.get("email", "")
        
        if not email:
            # Try alternative paths
            email = data.get("customer_email", "") or data.get("email", "")
        
        if not email:
            print(f"[subscription] Webhook missing email: {event_type}")
            return False
        
        subscription_id = data.get("subscription_id") or data.get("id")
        product_id = data.get("product_id", "")
        
        # Determine plan from product_id
        monthly_id = os.getenv("DODO_MONTHLY_PRODUCT_ID", "")
        yearly_id = os.getenv("DODO_YEARLY_PRODUCT_ID", "")
        
        if product_id == monthly_id:
            plan = "monthly"
        elif product_id == yearly_id:
            plan = "yearly"
        else:
            plan = "premium"
        
        # Handle based on event type
        if event_type in ("subscription.active", "subscription.renewed"):
            expires_at = data.get("current_period_end") or data.get("next_billing_date")
            # Clear cancelled_at when subscription is renewed/activated
            client = _get_supabase()
            client.table("subscriptions").upsert({
                "email": email.lower(),
                "is_premium": True,
                "plan": plan,
                "subscription_id": subscription_id,
                "expires_at": str(expires_at) if expires_at else None,
                "cancelled_at": None,  # Clear cancellation
                "updated_at": datetime.now().isoformat(),
            }, on_conflict="email").execute()
            clear_cache(email)
            print(f"[subscription] Activated: {email}")
            return True
            
        elif event_type in ("subscription.cancelled", "subscription.on_hold", "subscription.expired"):
            _save_subscription_to_db(email, False, None, subscription_id, None)
            clear_cache(email)
            print(f"[subscription] Deactivated: {email}")
            return True
        
        else:
            print(f"[subscription] Unhandled event: {event_type}")
            return True  # Don't fail on unknown events
            
    except Exception as e:
        print(f"[subscription] Webhook error: {e}")
        return False
