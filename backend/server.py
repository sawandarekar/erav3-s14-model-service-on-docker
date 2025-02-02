from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        logger.info(f"Received request with data: {data}")
        
        # Extract text and max_length from request
        if not data or 'text' not in data or 'max_length' not in data:
            error_msg = 'Missing required parameters: text and max_length'
            logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
            
        input_text = data['text']
        max_length = data['max_length']
        
        # Validate text
        if not input_text or not isinstance(input_text, str):
            error_msg = 'Text must be a non-empty string'
            logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
        
        # Validate max_length
        try:
            max_length = int(max_length)
            if max_length < 1 or max_length > 100:
                raise ValueError("max_length must be between 1 and 100")
        except (TypeError, ValueError) as e:
            error_msg = str(e)
            logger.error(f"max_length validation error: {error_msg}")
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
            
        # Process the text (placeholder for actual text generation logic)
        generated_text = f"Generated text based on input (max length: {max_length}): {input_text}"
        
        # Prepare successful response
        response = {
            'status': 'success',
            'data': {
                'input_text': input_text,
                'max_length': max_length,
                'generated_text': generated_text
            }
        }
        
        logger.info("Successfully processed request")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logger.error(f"Unexpected error: {error_msg}")
        return jsonify({
            'error': error_msg,
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'text-generation-api'
    })

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=True) 