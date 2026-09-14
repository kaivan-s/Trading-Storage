"""
Supabase Auth for the Flask API.

The UI signs in with Google via supabase-js and sends
`Authorization: Bearer <access_token>` on every /api call.
This module checks that token against Supabase Auth and, if
AUTH_ALLOWED_EMAILS is set, that the Google account is on the list.
"""

from __future__ import annotations

import hashlib
import os
import time

import requests
from flask import jsonify, request

import db

# Public: the Vite app and the authorize redirect need these.
# Cron endpoints use their own secret-based auth.
PUBLIC_PATHS = {"/api/auth/config", "/api/quote", "/api/cron/sector-lookouts", "/api/cron/coiled-bases"}

_CACHE_TTL = 45.0
_cache: dict[str, tuple[float, dict]] = {}


def anon_key() -> str:
    return (os.environ.get("SUPABASE_ANON_KEY") or "").strip()


def enabled() -> bool:
    """Auth is on only once the publishable/anon key is present."""
    return bool((db.SUPABASE_URL or "").strip() and anon_key())


def allowed_emails() -> set[str]:
    raw = os.environ.get("AUTH_ALLOWED_EMAILS") or ""
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _cache_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _verify(token: str) -> dict | None:
    now = time.monotonic()
    key = _cache_key(token)
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]

    url = (db.SUPABASE_URL or "").rstrip("/")
    api_key = db.SUPABASE_KEY or anon_key()
    if not url or not api_key or not token:
        return None

    try:
        r = requests.get(
            f"{url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": api_key,
            },
            timeout=8,
        )
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None
    user = r.json() or {}
    if not user.get("id"):
        return None
    _cache[key] = (now + _CACHE_TTL, user)
    if len(_cache) > 256:
        dead = [k for k, (exp, _) in _cache.items() if exp <= now]
        for k in dead:
            _cache.pop(k, None)
    return user


def current_user() -> tuple[dict | None, str | None, int]:
    header = request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        return None, "Sign in required", 401
    token = header.split(" ", 1)[1].strip()
    if not token:
        return None, "Sign in required", 401

    user = _verify(token)
    if not user:
        return None, "Invalid or expired session", 401

    email = (user.get("email") or "").strip().lower()
    allow = allowed_emails()
    if allow and email not in allow:
        return None, "This account is not allowed", 403
    return user, None, 200


def public_config() -> dict:
    return {
        "url": db.SUPABASE_URL or "",
        "anonKey": anon_key(),
        "allowlist": bool(allowed_emails()),
    }


def before_request():
    if request.method == "OPTIONS":
        return None
    path = request.path or ""
    if not path.startswith("/api/"):
        return None
    if path in PUBLIC_PATHS:
        return None
    if not enabled():
        return None

    user, err, code = current_user()
    if user is None:
        return jsonify({"error": err or "Sign in required"}), code
    request.environ["auth_user"] = user
    return None


def me_payload() -> dict:
    user = request.environ.get("auth_user") or {}
    meta = user.get("user_metadata") or {}
    email = user.get("email") or ""
    
    # Check subscription status
    try:
        from subscription import check_subscription
        sub = check_subscription(email) if email else {}
    except Exception:
        sub = {}
    
    return {
        "id": user.get("id"),
        "email": email,
        "name": meta.get("full_name") or meta.get("name") or email,
        "avatar": meta.get("avatar_url") or meta.get("picture"),
        "is_premium": sub.get("is_premium", False),
        "plan": sub.get("plan"),
        "subscription_expires": sub.get("expires_at"),
    }
