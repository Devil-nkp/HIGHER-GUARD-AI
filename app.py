import os
import io
import re
import cv2
import numpy as np
import pytesseract
import easyocr
import nltk
import spacy
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageEnhance
from nltk.sentiment import SentimentIntensityAnalyzer
from pdf2image import convert_from_bytes

# Initialize Flask
app = Flask(__name__)

# --- CONFIGURATION & SETUP ---

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Load Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize Analyzer
sia = SentimentIntensityAnalyzer()
reader = easyocr.Reader(['en'], gpu=False) # CPU for Render

# --- CORE LOGIC CLASS ---

class UltraStrongAnalyzer:
    def preprocess_image(self, image):
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. Rescaling
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 3. Thresholding (Otsu)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return [gray, denoised, thresh]

    def extract_text(self, file_bytes, filename):
        text_results = []
        
        try:
            # Handle PDF
            if filename.lower().endswith('.pdf'):
                images = convert_from_bytes(file_bytes)
                pil_images = images
            # Handle Images
            else:
                pil_image = Image.open(io.BytesIO(file_bytes))
                pil_images = [pil_image]

            for pil_img in pil_images:
                # Convert PIL to CV2
                open_cv_image = np.array(pil_img) 
                if len(open_cv_image.shape) == 3:
                    open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # 1. EasyOCR Extraction (Deep Learning)
                try:
                    ez_result = reader.readtext(open_cv_image, detail=0)
                    text_results.append(" ".join(ez_result))
                except:
                    pass

                # 2. Tesseract Extraction (Optical)
                processed_versions = self.preprocess_image(open_cv_image)
                config = '--oem 3 --psm 6'
                
                for img_ver in processed_versions:
                    try:
                        text = pytesseract.image_to_string(img_ver, config=config)
                        text_results.append(text)
                    except:
                        continue

            # Combine and clean text
            full_text = " ".join(text_results)
            return " ".join(full_text.split()) # Remove extra whitespace

        except Exception as e:
            print(f"Extraction Error: {e}")
            return ""

    def analyze_text(self, text):
        if len(text) < 50:
            return {
                "is_job_post": False,
                "error": "Not enough text detected to analyze."
            }

        text_lower = text.lower()
        
        # 1. Job Post Verification
        job_keywords = ['job', 'hiring', 'role', 'salary', 'resume', 'apply', 'vacancy', 'career', 'position', 'urgent']
        keyword_count = sum(1 for k in job_keywords if k in text_lower)
        
        if keyword_count < 2:
             return {"is_job_post": False, "extracted_text_preview": text[:500]}

        # 2. Risk Feature Extraction
        features = {
            'urgency_score': sum(text_lower.count(w) for w in ['urgent', 'immediate', 'asap', 'now']),
            'money_mentions': sum(text_lower.count(w) for w in ['$', 'salary', 'cash', 'earn', 'daily']),
            'sentiment': sia.polarity_scores(text)['compound'],
            'word_count': len(text.split()),
            'contact_info': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)) + 
                            len(re.findall(r'\d{10}', text))
        }

        # 3. Fraud Detection Rules
        risk_factors = []
        risk_score = 0

        # Pattern: High Urgency
        if features['urgency_score'] > 2:
            risk_factors.append("Suspiciously high urgency language")
            risk_score += 0.3

        # Pattern: Too good to be true / Money focus
        if features['money_mentions'] > 4 and features['word_count'] < 200:
            risk_factors.append("Excessive focus on money with little detail")
            risk_score += 0.4

        # Pattern: Gmail/Yahoo/Hotmail for business
        if re.search(r'@(gmail|yahoo|hotmail|outlook)\.com', text_lower):
            risk_factors.append("Unprofessional email domain (Gmail/Yahoo)")
            risk_score += 0.3

        # Pattern: Chat Apps
        if any(x in text_lower for x in ['whatsapp', 'telegram', 'dm me']):
            risk_factors.append("Request to connect via WhatsApp/Telegram")
            risk_score += 0.35

        # Pattern: No Interview
        if 'no interview' in text_lower or 'no experience' in text_lower:
            risk_factors.append("Claims of 'No Interview' or 'No Experience'")
            risk_score += 0.2

        # Pattern: Short Description
        if features['word_count'] < 80:
            risk_factors.append("Job description is suspiciously short")
            risk_score += 0.15

        # Cap score
        risk_score = min(risk_score, 0.99)
        if risk_score == 0: risk_score = 0.05 # Baseline

        is_fake = risk_score > 0.4

        return {
            "success": True,
            "is_job_post": True,
            "is_fake": is_fake,
            "risk_score": risk_score,
            "confidence": risk_score if is_fake else (1 - risk_score),
            "risk_factors": risk_factors if risk_factors else ["No major red flags detected"],
            "features": features,
            "extracted_text_preview": text[:1000] + "..."
        }

analyzer = UltraStrongAnalyzer()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})

    try:
        file_bytes = file.read()
        extracted_text = analyzer.extract_text(file_bytes, file.filename)
        result = analyzer.analyze_text(extracted_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Render provides PORT env var
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
