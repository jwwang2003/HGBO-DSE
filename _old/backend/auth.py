import redis
from itsdangerous import URLSafeTimedSerializer
from functools import wraps
from flask import request, jsonify, abort, make_response
import os

# Environment variables
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = os.getenv('REDIS_PORT')
SECRET_KEY = os.getenv('SECRET_KEY', '123456')
TOKEN_EXPIRATION = int(os.getenv('AUTH_TOKEN_EXP', 3600))  # Default: 1HR

# Auth token Redis cache
session_r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=int(os.getenv('AUTH_SESSION_MANAGER_REDIS_DB', '1'))
)

# Setup a URLSafeTimedSerializer to create and verify tokens
serializer = URLSafeTimedSerializer(SECRET_KEY)

def generate_auth_token(user_id):
    """
    Generate a token and store it in Redis
    """
    token = serializer.dumps(user_id)
    session_r.setex(token, TOKEN_EXPIRATION, user_id)  # Store token in Redis with expiration time
    return token

def verify_auth_token(token):
    """
    Verify token and check if it's expired  
    """
    user_id = session_r.get(token)
    if user_id is None:
        return None  # Token not found or expired
    try:
        # Check if the token is valid (not tampered with)
        user_id = serializer.loads(token, max_age=TOKEN_EXPIRATION)
    except Exception:
        return None  # Token is invalid or expired
    return user_id

def require_auth_token(f):
    """
    Decorator function to require authentication token.
    If the token is invalid or expired, it will be removed from the cookies.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('auth_token')  # Get token from cookies
        # Check if the token exists and is valid
        if not token or verify_auth_token(token) is None:
            # If the token is invalid or expired, clear the token from cookies
            resp = make_response(jsonify({"error": "Invalid or expired authentication token"}), 401)
            resp.set_cookie('auth_token', '', expires=0)  # Invalidate the cookie
            return resp
        return f(*args, **kwargs)
    return decorated
