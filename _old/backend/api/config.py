"""
backend/api/config.py
"""

from flask import Blueprint, jsonify
from bome.hls_dse import supported as dse_supported  # Import the required module

config_bp = Blueprint('config', __name__)

# Endpoint to return the valid configuration values
@config_bp.route('valid_options', methods=['GET'])
def get_valid_configuration_options():
    supported = {
        "mode": dse_supported['mode'],  # Access 'mode' from bome.hls_dse
        "alg": dse_supported['alg'],    # Access 'alg' from bome.hls_dse
        "device": dse_supported['device'],  # Access 'device' from bome.hls_dse
        "encode": dse_supported['encode'],  # Access 'encode' from bome.hls_dse
        "space": dse_supported['space']  # Access 'space' from bome.hls_dse
    }
    return jsonify(supported)