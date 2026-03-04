#  Fake Job Detection System – OCR + NLP Fraud Analyzer

An AI-powered system that detects fraudulent job postings from PDFs and images using OCR, NLP, and risk-based classification.

---

##  Overview

Online job fraud is increasing rapidly. This system analyzes job documents and identifies suspicious patterns such as:

- Excessive urgency language
- Overemphasis on money
- Unprofessional email domains
- Requests to contact via WhatsApp/Telegram
- "No Interview / No Experience" claims

The system extracts text using OCR and performs NLP-based risk analysis to classify postings as potentially fake.

---

##  Architecture

Document Upload (PDF / Image)  
→ OCR Preprocessing (OpenCV)  
→ Text Extraction (Tesseract OCR)  
→ Feature Engineering  
→ Sentiment Analysis (VADER)  
→ Risk Scoring Engine  
→ Fraud Classification + Confidence Score  

---

##  Tech Stack

- Python
- Flask
- OpenCV
- Tesseract OCR
- NLTK (VADER)
- spaCy
- NumPy
- Regex
- pdf2image

---

##  Core Features

- Supports PDF and Image inputs
-  Advanced image preprocessing (rescaling, denoising, Otsu thresholding)
-  Sentiment analysis + linguistic parsing
-  Custom fraud risk scoring system
-  Real-time REST API analysis

---

##  Risk Factors Detected

- High urgency keywords
- Money-focused short descriptions
- Unprofessional email domains
- Chat app contact requests
- Suspiciously short job descriptions

---

##  Installation

```bash
git clone https://github.com/Devil-nkp/Fake-Job-Detection.git
cd Fake-Job-Detection
pip install -r requirements.txt
```

Install Tesseract:

- Windows: Install Tesseract and add to PATH
- Linux:
```bash
sudo apt install tesseract-ocr
```

Run the application:

```bash
python app.py
```

Access LINK:
```
https://higher-guard-ai.onrender.com/
```

---

##  Future Improvements

- Train ML classifier (Logistic Regression / Random Forest)
- Add dataset-based evaluation metrics (Accuracy, F1-score)
- Integrate database logging
- Add frontend dashboard

---

##  Author

Naveenkumar G  
AI / ML Engineer
