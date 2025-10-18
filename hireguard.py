!pip install gradio tensorflow opencv-python pytesseract Pillow PyPDF2 pdf2image spacy nltk scikit-learn matplotlib seaborn wordcloud easyocr
!sudo apt install tesseract-ocr libtesseract-dev
!pip install pytesseract pillow imutils
!python -m spacy download en_core_web_sm

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import PyPDF2
import pdf2image
import re
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import tempfile
import os
import io
import base64
import imutils
import easyocr
from google.colab import files
warnings.filterwarnings('ignore')

# Downloading NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Set path for Tesseract in Colab
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Loading spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("spaCy model not found. Using limited NLP features.")
    nlp = None

class UltraStrongOCRProcessor:
    def __init__(self):
        self.tesseract_configs = [
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()-_=+[]{};:\'"\\|<>/ ',
            '--oem 3 --psm 1',
            '--oem 3 --psm 3',
            '--oem 3 --psm 4',
            '--oem 3 --psm 8',
            '--oem 3 --psm 11',
            '--oem 1 --psm 6',  # Legacy engine
            '--oem 3 --psm 12',  # Sparse text
        ]
        # Initialize EasyOCR reader
        self.easyocr_reader = easyocr.Reader(['en'])

    def super_preprocess_image(self, image_path):
        """Apply multiple advanced preprocessing techniques to enhance OCR accuracy"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply multiple preprocessing techniques
            processed_images = []

            # 1. Resize image for better processing (if too small)
            height, width = gray.shape
            if max(height, width) < 1000:
                scale_factor = 2000 / max(height, width)
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # 2. Noise reduction with multiple techniques
            denoised1 = cv2.fastNlMeansDenoising(gray)
            denoised2 = cv2.medianBlur(gray, 3)
            denoised3 = cv2.GaussianBlur(gray, (3, 3), 0)

            processed_images.extend([denoised1, denoised2, denoised3])

            # 3. Multiple thresholding techniques
            # Otsu's thresholding
            _, thresh_otsu = cv2.threshold(denoised1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Adaptive thresholding
            thresh_adapt1 = cv2.adaptiveThreshold(denoised1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
            thresh_adapt2 = cv2.adaptiveThreshold(denoised1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
            thresh_adapt3 = cv2.adaptiveThreshold(denoised1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 21, 7)

            processed_images.extend([thresh_otsu, thresh_adapt1, thresh_adapt2, thresh_adapt3])

            # 4. Morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)

            processed_images.extend([opening, closing])

            # 5. CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(denoised1)
            processed_images.append(clahe_img)

            # 6. Edge enhancement
            edges = cv2.Canny(denoised1, 100, 200)
            processed_images.append(edges)

            # 7. PIL-based enhancements
            pil_image = Image.fromarray(gray)
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_image)
            contrast_img = enhancer.enhance(2.0)
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(contrast_img)
            sharp_img = enhancer.enhance(2.0)
            # Convert back to numpy array
            pil_enhanced = np.array(sharp_img)
            processed_images.append(pil_enhanced)

            # 8. Bilateral filter for edge-preserving smoothing
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            processed_images.append(bilateral)

            return processed_images

        except Exception as e:
            print(f"Error in super preprocessing: {e}")
            return None

    def extract_text_with_all_engines(self, image_path):
        """Extract text using multiple OCR engines and techniques"""
        all_texts = []

        # Preprocess image with multiple techniques
        processed_images = self.super_preprocess_image(image_path)
        if processed_images is None:
            return None

        # Try Tesseract OCR on each preprocessed image with multiple configurations
        for i, processed_img in enumerate(processed_images):
            for config in self.tesseract_configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text and len(text.strip()) > 5:  # Lower threshold to catch more text
                        all_texts.append(text)
                except:
                    continue

        # Try EasyOCR
        try:
            easyocr_results = self.easyocr_reader.readtext(image_path, detail=0)
            easyocr_text = " ".join(easyocr_results)
            if easyocr_text and len(easyocr_text.strip()) > 5:
                all_texts.append(easyocr_text)
        except:
            pass

        # Also try with original image using both engines
        original_image = cv2.imread(image_path)
        if original_image is not None:
            # Tesseract on original
            for config in self.tesseract_configs:
                try:
                    text = pytesseract.image_to_string(original_image, config=config)
                    if text and len(text.strip()) > 5:
                        all_texts.append(text)
                except:
                    continue

            # EasyOCR on original
            try:
                easyocr_results = self.easyocr_reader.readtext(image_path, detail=0)
                easyocr_text = " ".join(easyocr_results)
                if easyocr_text and len(easyocr_text.strip()) > 5:
                    all_texts.append(easyocr_text)
            except:
                pass

        # Remove duplicates and empty texts
        unique_texts = [t for t in set(all_texts) if len(t.strip()) > 5]

        if unique_texts:
            # Combine all texts and clean up
            combined_text = "\n".join(unique_texts)

            # Post-processing to clean up the text
            cleaned_text = self.post_process_text(combined_text)

            return cleaned_text

        return None

    def post_process_text(self, text):
        """Clean and post-process extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors
        replacements = {
            r'(\w)\.(\w)': r'\1. \2',  # Add space after periods between words
            r'\,(\w)': r', \1',        # Add space after commas
            r'\;(\w)': r'; \1',        # Add space after semicolons
            r'\:(\w)': r': \1',        # Add space after colons
            r'\!(\w)': r'! \1',        # Add space after exclamation marks
            r'\?(\w)': r'? \1',        # Add space after question marks
            r'\s+\.': '.',             # Remove spaces before periods
            r'\s+\,': ',',             # Remove spaces before commas
            r'i\.e\.': 'i.e.',         # Fix i.e.
            r'e\.g\.': 'e.g.',         # Fix e.g.
            r'(\w)-\s+(\w)': r'\1-\2', # Fix hyphenated words
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Capitalize sentences
        sentences = re.split(r'([.!?])\s+', text)
        text = ''
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i+1] if i+1 < len(sentences) else ''
            if sentence:
                text += sentence[0].upper() + sentence[1:] + punctuation + ' '

        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Enhanced PDF text extraction with multiple methods"""
        all_text = []

        try:
            # Method 1: Try pdf2image + OCR with high quality
            images = pdf2image.convert_from_path(pdf_path, dpi=400, thread_count=4)
            for img in images:
                # Save image temporarily and process
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img.save(tmp.name, 'JPEG', quality=100)
                    text = self.extract_text_with_all_engines(tmp.name)
                    if text:
                        all_text.append(text)
                    os.unlink(tmp.name)

            # Method 2: Try PyPDF2 for text extraction
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            all_text.append(text)
            except:
                pass

            # Method 3: Try pdfminer for better text extraction
            try:
                from pdfminer.high_level import extract_text
                pdf_text = extract_text(pdf_path)
                if pdf_text and len(pdf_text.strip()) > 10:
                    all_text.append(pdf_text)
            except:
                pass

            # Combine all extracted text
            if all_text:
                combined_text = "\n".join(all_text)
                return self.post_process_text(combined_text)

        except Exception as e:
            print(f"Error extracting from PDF: {e}")

        return None

class AdvancedJobPostAnalyzer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.sia = SentimentIntensityAnalyzer()
        self.ocr_processor = UltraStrongOCRProcessor()

    def extract_text_from_image(self, image_path):
        """Extract text from image using ultra-strong OCR"""
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return self.ocr_processor.extract_text_with_all_engines(image_path)
        elif image_path.lower().endswith('.pdf'):
            return self.ocr_processor.extract_text_from_pdf(image_path)
        return None

    def extract_text_from_bytes(self, file_bytes, file_extension):
        """Extract text from file bytes (for Gradio uploads)"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            # Extract text using the file path method
            text = self.extract_text_from_image(tmp_file_path)

            # Clean up
            os.unlink(tmp_file_path)

            return text

        except Exception as e:
            print(f"Error extracting text from bytes: {e}")
            return None

    def is_job_post(self, text):
        """Check if the extracted text appears to be a job posting"""
        if not text or len(text.strip()) < 50:
            return False

        text_lower = text.lower()
        job_keywords = [
            'job', 'position', 'hire', 'hiring', 'apply', 'application', 'role', 'career', 'vacancy',
            'responsibilities', 'requirements', 'qualifications', 'experience', 'salary', 'compensation',
            'company', 'employer', 'opening', 'opportunity', 'remote', 'full-time', 'part-time',
            'internship', 'contract', 'permanent', 'submit', 'resume', 'cv', 'interview'
        ]

        keyword_count = sum(1 for keyword in job_keywords if keyword in text_lower)

        # Require at least 3 job-related keywords for classification as job post
        # Also check for absence of non-job indicators (e.g., recipes, ads, etc.)
        non_job_indicators = ['recipe', 'ingredients', 'cooking', 'sale', 'buy', 'product', 'advertisement', 'news']
        non_job_count = sum(1 for indicator in non_job_indicators if indicator in text_lower)

        return keyword_count >= 3 and non_job_count < 2

    def extract_features(self, text):
        """Extract advanced features from job post text"""
        features = {}

        if not text:
            # Return default values if no text
            features['text_length'] = 0
            features['word_count'] = 0
            features['char_count'] = 0
            features['sentiment_compound'] = 0
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 0
            features['sentiment_neutral'] = 0
            features['uppercase_ratio'] = 0
            features['exclamation_count'] = 0
            features['question_mark_count'] = 0
            features['urgency_score'] = 0
            features['money_mentions'] = 0
            features['requirement_mentions'] = 0
            features['email_count'] = 0
            features['phone_count'] = 0
            features['noun_count'] = 0
            features['verb_count'] = 0
            features['adjective_count'] = 0
            features['entity_count'] = 0
            return features

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(" ", ""))

        # Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_neutral'] = sentiment['neu']

        # Linguistic features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['exclamation_count'] = text.count('!')
        features['question_mark_count'] = text.count('?')

        # Specific job post indicators
        urgency_phrases = ['urgent', 'immediate', 'quick', 'fast', 'ASAP', 'right away', 'hiring immediately', 'start immediately']
        features['urgency_score'] = sum(text.lower().count(phrase) for phrase in urgency_phrases)

        money_phrases = ['$', 'salary', 'pay', 'compensation', 'bonus', 'commission', 'earn', 'income', 'wage']
        features['money_mentions'] = sum(text.lower().count(phrase) for phrase in money_phrases)

        requirement_phrases = ['require', 'must have', 'necessary', 'qualification', 'experience', 'skill', 'education', 'degree']
        features['requirement_mentions'] = sum(text.lower().count(phrase) for phrase in requirement_phrases)

        # Email and phone patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b'

        features['email_count'] = len(re.findall(email_pattern, text))
        features['phone_count'] = len(re.findall(phone_pattern, text))

        # NLP features if spaCy is available
        if nlp and len(text) > 10:
            try:
                doc = nlp(text[:50000])  # Limit text length for processing
                features['noun_count'] = sum(1 for token in doc if token.pos_ == 'NOUN')
                features['verb_count'] = sum(1 for token in doc if token.pos_ == 'VERB')
                features['adjective_count'] = sum(1 for token in doc if token.pos_ == 'ADJ')
                features['entity_count'] = len(doc.ents)
            except:
                features['noun_count'] = 0
                features['verb_count'] = 0
                features['adjective_count'] = 0
                features['entity_count'] = 0
        else:
            features['noun_count'] = 0
            features['verb_count'] = 0
            features['adjective_count'] = 0
            features['entity_count'] = 0

        return features

    def analyze_job_post_from_bytes(self, file_bytes, file_extension):
        """Main method to analyze job post from file bytes"""
        # Extract text using ultra-strong OCR
        extracted_text = self.extract_text_from_bytes(file_bytes, file_extension)

        # If first attempt fails, try with enhanced processing
        if not extracted_text or len(extracted_text.strip()) < 20:
            try:
                # Convert bytes to PIL Image and apply super enhancement
                if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    image = Image.open(io.BytesIO(file_bytes))

                    # Multiple enhancement techniques
                    # 1. Resize if too small
                    width, height = image.size
                    if max(width, height) < 1000:
                        scale_factor = 2000 / max(width, height)
                        new_size = (int(width * scale_factor), int(height * scale_factor))
                        image = image.resize(new_size, Image.LANCZOS)

                    # 2. Enhance contrast and sharpness
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.5)
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(3.0)
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(1.2)

                    # 3. Convert to numpy array for OpenCV processing
                    cv_image = np.array(image)
                    if len(cv_image.shape) == 3:
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

                    # 4. Save enhanced image and try OCR again
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        cv2.imwrite(tmp_file.name, cv_image)
                        extracted_text = self.extract_text_from_image(tmp_file.name)
                        os.unlink(tmp_file.name)
            except Exception as e:
                print(f"Enhanced processing failed: {e}")

        if not extracted_text or len(extracted_text.strip()) < 20:
            return {
                'is_job_post': False,
                'is_fake': 'Unknown',
                'confidence': 0.0,
                'error': 'OCR failed to extract sufficient text. Please try with a clearer image or PDF.',
                'extracted_text': 'Text extraction failed. The ima