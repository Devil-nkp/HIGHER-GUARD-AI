import os

class Config:
    # Model paths
    MODEL_PATH = os.path.join('models', 'job_authenticity_predictor_model.h5')
    BERT_MODEL_NAME = 'bert-base-uncased'
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    
    # Text processing
    MAX_TEXT_LENGTH = 128
    MIN_WORD_COUNT_FOR_JOB_POST = 5
    
    # Server configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # File upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Logging
    LOG_FILE = 'app.log'
    LOG_LEVEL = 'INFO'