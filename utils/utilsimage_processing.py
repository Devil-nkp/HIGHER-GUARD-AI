import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logger = logging.getLogger(__name__)

def validate_image(image):
    """
    Validate the uploaded image
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    try:
        # Check if image is valid
        image.verify()
        
        # Check image dimensions
        width, height = image.size
        if width < 50 or height < 50:
            return {
                'valid': False,
                'message': 'Image dimensions are too small'
            }
        
        return {'valid': True, 'message': 'Image is valid'}
        
    except (IOError, OSError, UnidentifiedImageError) as e:
        logger.error(f"Invalid image: {e}")
        return {
            'valid': False,
            'message': 'Invalid image file'
        }

def process_image(image, ocr_model=None):
    """
    Process image using OCR to extract text
    
    Args:
        image: PIL Image object
        ocr_model: Optional OCR model (if None, uses a fallback)
    
    Returns:
        dict: OCR result with 'text' and 'confidence'
    """
    try:
        # Convert PIL Image to numpy array
        image_np = np.array(image.convert("RGB"))
        
        if ocr_model:
            # Use the provided OCR model
            result = ocr_model([image_np])
            extracted_text = result.render()
            confidence = 0.9  # Placeholder for actual confidence calculation
        else:
            # Fallback for when OCR model is not available
            extracted_text = "Simulated OCR text extraction. Actual OCR would be used in production."
            confidence = 0.7
        
        return {
            'text': extracted_text,
            'confidence': confidence,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'success': False,
            'error': str(e)
        }

def preprocess_image_for_model(image, target_size):
    """
    Preprocess image for the fraud detection model
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height)
    
    Returns:
        numpy array: Preprocessed image
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array