import re
import logging
import numpy as np
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

def analyze_text(text, image, model, tokenizer, config):
    """
    Analyze text using the fraud detection model
    
    Args:
        text: Extracted text from OCR
        image: Original PIL Image object
        model: Loaded fraud detection model
        tokenizer: Text tokenizer
        config: Configuration object
    
    Returns:
        dict: Analysis results with fraud probabilities
    """
    try:
        # Preprocess the image
        from .image_processing import preprocess_image_for_model
        image_tensor = preprocess_image_for_model(image, config.IMAGE_SIZE)
        
        # Tokenize the text
        tokenized = tokenizer(
            text, 
            max_length=config.MAX_TEXT_LENGTH, 
            truncation=True, 
            padding='max_length', 
            return_tensors='np'
        )
        
        # Prepare model inputs
        model_inputs = {
            'image_input': image_tensor,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        
        # Make prediction
        prediction = model.predict(model_inputs)[0][0]
        
        return {
            'FAKE': float(prediction),
            'REAL': 1 - float(prediction)
        }
        
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        # Fallback to rule-based analysis
        return calculate_fraud_score(text)

def calculate_fraud_score(text):
    """
    Calculate fraud score using rule-based approach
    (Fallback when AI model is not available)
    
    Args:
        text: Text to analyze
    
    Returns:
        dict: Fraud probability scores
    """
    text_lower = text.lower()
    fraud_score = 0
    
    # 1. Urgency indicators
    urgency_words = ['urgent', 'immediate', 'asap', 'hiring now', 'start today', 'quick hire']
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    fraud_score += min(20, urgency_count * 5)
    
    # 2. Personal email domains
    personal_domains = ['gmail.', 'yahoo.', 'hotmail.', 'outlook.', 'aol.']
    if any(domain in text_lower for domain in personal_domains):
        fraud_score += 15
    
    # 3. Unrealistic salary mentions
    salary_patterns = [r'\$\d{3},\d{3}', r'\$\d{2,3}k', r'\$\d{2,3},\d{3}']
    for pattern in salary_patterns:
        if re.search(pattern, text):
            fraud_score += 25
            break
    
    # 4. Requests for sensitive information
    sensitive_info = ['social security', 'ssn', 'bank account', 'credit card', 'passport', 'driver license']
    if any(info in text_lower for info in sensitive_info):
        fraud_score += 30
    
    # 5. Fees mentioned
    if 'fee' in text_lower or 'payment' in text_lower or 'charge' in text_lower:
        fraud_score += 20
    
    # 6. No experience required for high-paying jobs
    if ('no experience' in text_lower or 'no degree' in text_lower) and \
       ('senior' in text_lower or 'manager' in text_lower or 'director' in text_lower):
        fraud_score += 25
    
    # 7. Poor grammar and spelling
    grammar_errors = len(re.findall(r'\b(their|they\'re|there|your|you\'re|its|it\'s)\b', text_lower, re.IGNORECASE))
    fraud_score += min(15, grammar_errors * 3)
    
    # Cap at 100
    fraud_score = min(100, fraud_score)
    
    # Convert to probability
    fraud_probability = fraud_score / 100
    
    return {
        'FAKE': float(fraud_probability),
        'REAL': 1 - float(fraud_probability)
    }

def extract_job_details(text):
    """
    Extract key job details from text
    
    Args:
        text: Job posting text
    
    Returns:
        dict: Extracted job details
    """
    details = {
        'title': None,
        'company': None,
        'location': None,
        'salary': None,
        'requirements': []
    }
    
    # Simple pattern matching for extraction
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        # Extract job title
        if not details['title'] and any(keyword in line_lower for keyword in ['title', 'position', 'role']):
            details['title'] = line.strip()
        
        # Extract company
        if not details['company'] and 'company' in line_lower:
            details['company'] = line.strip()
        
        # Extract location
        if not details['location'] and any(keyword in line_lower for keyword in ['location', 'remote', 'onsite', 'hybrid']):
            details['location'] = line.strip()
        
        # Extract salary
        if not details['salary'] and any(keyword in line_lower for keyword in ['salary', 'pay', 'compensation']):
            details['salary'] = line.strip()
        
        # Extract requirements
        if 'requirement' in line_lower or 'qualification' in line_lower:
            details['requirements'].append(line.strip())
    
    return details