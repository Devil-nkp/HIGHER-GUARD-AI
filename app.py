import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import BertTokenizer, TFBertModel
from doctr.models import ocr_predictor
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class Config:
    MODEL_PATH = './models/job_authenticity_predictor_model.h5'
    IMAGE_SIZE = (224, 224)
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LENGTH = 128
    MIN_WORD_COUNT_FOR_JOB_POST = 5
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    UPLOAD_FOLDER = './uploads'

class BertLayer(tf.keras.layers.Layer):
    """Custom Keras layer for BERT model integration"""
    def __init__(self, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert.trainable = False
    
    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    
    def get_config(self):
        return super(BertLayer, self).get_config()

# Initialize AI components
logger.info("Initializing AI components...")
config = Config()

# Ensure upload folder exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Load Fraud Detection Model
fraud_model = None
try:
    with tf.keras.utils.custom_object_scope({'BertLayer': BertLayer}):
        fraud_model = tf.keras.models.load_model(config.MODEL_PATH)
    logger.info("✅ Fraud Detection Model loaded successfully")
except Exception as e:
    logger.error(f"🚨 Failed to load fraud detection model: {str(e)}")
    raise

# Initialize Document AI OCR Predictor
logger.info("Initializing Document AI Engine (doctr)...")
ocr_predictor_model = ocr_predictor(pretrained=True)
logger.info("✅ Document AI Engine ready")

# Initialize Text Tokenizer
tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
logger.info("✅ Text Tokenizer ready")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def document_intelligence_pipeline(image_path: str) -> Tuple[str, Dict]:
    """
    3-Stage Document Intelligence Pipeline:
    1. Document AI OCR
    2. Content Sanity Check
    3. Multimodal Fraud Analysis
    """
    if not fraud_model:
        return "ERROR: Model not loaded", {"ERROR": 1.0}

    try:
        # Open and convert image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # --- STAGE 1: DOCUMENT AI OCR ---
        logger.info("Running Document AI OCR...")
        result = ocr_predictor_model([image_np])
        extracted_text = result.render()
        logger.info("OCR completed successfully")
        
        # --- STAGE 2: CONTENT SANITY CHECK ---
        word_count = len(extracted_text.split())
        if word_count < config.MIN_WORD_COUNT_FOR_JOB_POST:
            logger.warning(f"Content sanity check failed - only {word_count} words found")
            return "Invalid content - not a job posting", {"INVALID": 1.0}
        
        logger.info(f"Content sanity check passed - {word_count} words found")
        
        # --- STAGE 3: MULTIMODAL FRAUD ANALYSIS ---
        logger.info("Running Fraud Detection Model...")
        
        # Preprocess image
        image_for_model = image.resize(config.IMAGE_SIZE)
        image_tensor = np.expand_dims(np.array(image_for_model), axis=0)
        
        # Tokenize text
        tokenized = tokenizer(
            extracted_text, 
            max_length=Config.MAX_TEXT_LENGTH, 
            truncation=True, 
            padding='max_length', 
            return_tensors='tf'
        )
        
        # Prepare model inputs
        model_inputs = {
            'image_input': image_tensor,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        
        # Get prediction
        prediction_score = fraud_model.predict(model_inputs)[0][0]
        prediction_dict = {
            'FAKE': float(prediction_score), 
            'REAL': 1 - float(prediction_score)
        }
        
        logger.info(f"Fraud analysis completed: {prediction_dict}")
        return extracted_text, prediction_dict
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {str(e)}")
        return f"ERROR: {str(e)}", {"ERROR": 1.0}

@app.route('/api/analyze', methods=['POST'])
def analyze_job_posting():
    """API endpoint for job posting analysis"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process the file
            extracted_text, prediction = document_intelligence_pipeline(filepath)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {str(e)}")
            
            # Prepare response
            response = {
                "extracted_text": extracted_text,
                "prediction": prediction,
                "is_job_posting": len(extracted_text.split()) >= config.MIN_WORD_COUNT_FOR_JOB_POST
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "model_loaded": fraud_model is not None,
        "ocr_ready": ocr_predictor_model is not None,
        "tokenizer_ready": tokenizer is not None,
        "status": "OK" if fraud_model else "ERROR"
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)