import datetime
import io
import time
import uuid
import logging
from typing import Dict, Tuple, Optional
from PIL import Image
from flask import Flask, request, jsonify
from config import Config
from model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)%s - [%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

config = Config()
model_loader = None

def init_model():
    '''
    Initialize and load ML model.

    Implement model initialization
    - Create ModelLoader instance
    - Load model
    - Handle errors gracefully
    - This is called once at startup
    '''
    try:
        logger.info("Initializing model...")
        model_loader = ModelLoader(
            model_name = config.MODEL_NAME,
            device = config.DEVICE
        )
        model_loader.load()
        logger.info("Model initialized sccessfully.")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise #主动抛出一个错误，让程序停下来

    return True

@app.route('/health', methods=['GET'])
def health():
    '''
    Health check endpoint.

    Implement health check
    - Check if model is loaded
    - Return healthy status if model loaded
    - Return unhealthy (503) if model not loaded
    - Include model name and uptime
    Returns:
        JSON response with health status
    '''
    is_healthy = (model_loader is not None) and (model_loader.model is not None)

    if is_healthy:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_name': config.MODEL_NAME,
            'uptime': time.time()
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'reason': 'Model not loaded',
            'uptime': time.time()
        }), 503

@app.route('/info', methods=['GET'])
def info():
    '''
    Model information endpoint.

    Implement info endpoint
    - Return model metadata
    - Include API version
    - Include supported endpoints
    - Include limits (file size, timeout, etc.)

    Returns:
        JSON response with model and API info
    '''
    if model_loader is None:
        return jsonify({
            'error': 'Model not loaded',
            'uptime': time.time()
        }),503
    model_info = model_loader.get_model_info()

    return jsonify({
        'model': model_info,
        'api' : {
            'version': config.API_VERSION,
            'endpoints': ['/predict', '/health', '/info'],
        },
        'limits': {
            'max_file_size_mb': config.MAX_FILE_SIZE / (1024 * 1024),
            'max_image_dimension': config.NAX_IMAGE_DIMENSION,
            'timeout_seconds' : config.REQUEST_TIMEOUT
        },
        'uptime': time.time()
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Prediction endpoint.

    Implement prediction endpoint
    1. Generate correlation ID for request tracking
    2. Validate request has file
    3. Validate file size
    4. Load and validate image
    5. Get top_k parameter (optional)
    6. Call model_loader.predict()
    7. Measure latency
    8. Format response
    9. Log request
    10. Handle all errors gracefully

    Returns:
        JSON response with predictions or error
    '''
    correlation_id = generate_correlation_id()
    start_time = time.time()
    try:
        if 'file' not in request.files:
            return format_error_response({
                'MISSING_FILE',
    #           'No file provided in request',
    #            correlation_id
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file name found in request',
                'correlation_id': correlation_id
            }), 400

        if file.content_length > config.MAX_FILE_SIZE:
            pass
    except Exception as e:
        pass


# =========================================================================
# Helper Functions
# =========================================================================

def generate_correlation_id() -> str:
    '''
    Generate unique correlation ID for request tracking.

    Implement correlation ID generation
    - Use UUID for uniqueness
    - Format as 'req-<8-char-hex>'
    - Used for tracing requests in logs

    Returns:
        Correlation ID string
    '''
    return str(uuid.uuid4())

def format_error_response(error_code: str, message: str, correlation_id: str, details: Optional[dict] = None) -> dict:
    '''
    Format error response.

    TODO: Implement error response formatting
    - Include success=False
    - Include error object with code, message
    - Include correlation_id for tracking
    - Include timestamp
    - Optionally include details

    Args:
        error_code: Error code (e.g., 'INVALID_IMAGE')
        message: Human-readable error message
        correlation_id: Request correlation ID
        details: Optional additional details

    Returns:
        Formatted error response dictionary
    '''
    error_response = {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    }

    if details:
        error_response['error']['details'] = details

    return error_response

def validate_image(file) -> Tuple[bool, Optional[str]]:
    '''
    Validate uploaded file is a valid image.

    Implement file validation
    - Check file is not None
    - Check file has content
    - Try to open with PIL
    - Return (is_valid, error_message)

    Args:
        file: Flask file object

    Returns:
        Tuple of (is_valid, error_message)
    '''
    pass
if __name__ == '__main__':
    pass