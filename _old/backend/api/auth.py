
"""
backend/api/auth.py

Notes:
- Authentication token is saved at `auth_token` as a cookie
"""

from flask import Blueprint, request, jsonify, make_response
from backend.auth import session_r, generate_auth_token, verify_auth_token, require_auth_token
import os
import uuid

MASTER_API_KEY = os.getenv('MASTER_API_KEY', '123456')
TOKEN_EXPIRATION = os.getenv('AUTH_TOKEN_EXP', 3600)

auth_bp = Blueprint('auth', __name__)

# Authentication: Login and token generation
@auth_bp.route('/login', methods=['POST'])
def login():
    # Check if the user is already logged in (token exists in cookies)
    token = request.cookies.get('auth_token')
    if token:
        user_id = verify_auth_token(token)  # Check if the token is valid
        if user_id:
            return jsonify(
                { 'message': "Already logged in", 'data': { 'user_id': user_id } }
            ), 200
        else:
            # If the token is invalid, invalidate the cookie
            resp = make_response(jsonify({"error": "Invalid token, please log in again"}))
            resp.set_cookie('auth_token', '', expires=0)  # Invalidate the cookie
            return resp, 403
    
    data = request.get_json()
    api_key = data.get('api_key')
    
    if api_key != MASTER_API_KEY:
        return jsonify({"error": "Invalid API Key"}), 403
    
    # Generate a new UUID for the user if they don't already have one
    user_id = str(uuid.uuid4())  # Generate a unique user ID as UUID
    
    # Generate the authentication token using the user_id
    token = generate_auth_token(user_id)
    
    # Create a response and set the token as a cookie, and also return the user_id in the response
    resp = make_response(
        jsonify(
            { 'message': "Login successful", 'data': { 'user_id': user_id } }
        )
    )
    resp.set_cookie('auth_token', token, max_age=None, httponly=True)
    
    return resp

# Logout endpoint
@auth_bp.route('/logout', methods=['POST'])
@require_auth_token
def logout():
    token = request.cookies.get('auth_token')  # Get token from cookies
    # Remove the token from Redis if it exists
    if token:
        session_r.delete(token)  # Delete the token from Redis
    resp = make_response(jsonify({'message': "Logged out successfully"}))
    resp.set_cookie('auth_token', '', expires=0)  # Remove the token from the cookie
    return resp