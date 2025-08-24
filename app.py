import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from doctr.models import ocr_predictor
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from io import BytesIO
import json
from datetime import datetime
from utils.image_processing import process_image, validate_image
from utils.text_processing import analyze_text, calculate_fraud_score
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
class Config:
    MODEL_PATH = 'models/job_authenticity_predictor_model.h5'
    IMAGE_SIZE = (224, 224)
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LENGTH = 128
    MIN_WORD_COUNT_FOR_JOB_POST = 5
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert.trainable = False
    
    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    
    def get_config(self):
        return super(BertLayer, self).get_config()
fraud_model = None
ocr_model = None
tokenizer = None

def load_models():
    global fraud_model, ocr_model, tokenizer
    
    try:
        if os.path.exists(Config.MODEL_PATH):
            with tf.keras.utils.custom_object_scope({'BertLayer': BertLayer}):
                fraud_model = tf.keras.models.load_model(Config.MODEL_PATH)
            logger.info("Fraud Detection Model loaded successfully")
        else:
            logger.warning("Fraud detection model not found")
        
        ocr_model = ocr_predictor(pretrained=True)
        logger.info("OCR Model loaded successfully")
        
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': fraud_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_job_posting():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            image = Image.open(file.stream).convert('RGB')
            validation_result = validate_image(image)
            
            if not validation_result['valid']:
                return jsonify({
                    'error': validation_result['message'],
                    'extracted_text': '',
                    'analysis': {'INVALID_IMAGE': 1.0}
                }), 400
            
            logger.info("Starting OCR processing")
            ocr_result = process_image(image, ocr_model)
            extracted_text = ocr_result['text']
            
            word_count = len(extracted_text.split())
            if word_count < Config.MIN_WORD_COUNT_FOR_JOB_POST:
                return jsonify({
                    'error': 'The uploaded content does not appear to be a job posting',
                    'extracted_text': extracted_text,
                    'analysis': {'INVALID_CONTENT': 1.0}
                }), 400
            
            logger.info("Starting fraud analysis")
            if fraud_model and tokenizer:
                analysis_result = analyze_text(extracted_text, image, fraud_model, tokenizer, Config)
            else:
                analysis_result = calculate_fraud_score(extracted_text)
            
            logger.info(f"Analysis complete: {analysis_result}")
            
            return jsonify({
                'success': True,
                'extracted_text': extracted_text,
                'analysis': analysis_result,
                'word_count': word_count
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text_only():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        word_count = len(text.split())
        if word_count < Config.MIN_WORD_COUNT_FOR_JOB_POST:
            return jsonify({
                'error': 'The text does not appear to be a job posting',
                'analysis': {'INVALID_CONTENT': 1.0}
            }), 400
        
        analysis_result = calculate_fraud_score(text)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'word_count': word_count
        })
        
    except Exception as e:
        logger.error(f"Error processing text analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        
        if not data or 'rating' not in data:
            return jsonify({'error': 'No rating provided'}), 400
        
        logger.info(f"User feedback: {data}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback received. Thank you!'
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Loading AI models...")
    load_models()
    
    logger.info("Starting HireGuard AI server...")
    app.run(host='0.0.0.0', port=5000, debug=True)