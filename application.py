"""
WSGI entry for Elastic Beanstalk / gunicorn.

EB defaults to `application:application`. Local runs still use
`python server.py`.
"""

from server import app as application, _boot
import threading

threading.Thread(target=_boot, daemon=True).start()
