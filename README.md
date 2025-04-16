# 📧 NLP Spam Classifier

A Flask web application that detects whether an email is **Spam** or **Not Spam** using pretrained machine learning models. Users can either **paste the email text** or **upload an image** of the email — the app extracts text using **OCR (Tesseract)** and classifies it.

🔗 **Live Repo:** [github.com/lalit-29r/NLP-Spam-Classifier](https://github.com/lalit-29r/NLP-Spam-Classifier)

---

## 🚀 Features

- 🧠 Pretrained KNN and Random Forest models
- ✍️ Text input and 📷 Image upload with OCR
- 📤 Email text extraction and classification
- 💻 Clean and responsive UI
- 🐍 Built with Python, Flask, scikit-learn, pytesseract

---

## 📂 Project Structure

```
NLP-Spam-Classifier/
├── app.py
├── model_knn.pkl
├── model_rf.pkl
├── preprocessing.py
├── requirements.txt
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── README.md
```

---

## 🛠 Requirements

- Python 3.6+
- Flask
- scikit-learn
- pytesseract
- Pillow
- Tesseract-OCR (installed & added to system PATH)

---

## ⚙️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/lalit-29r/NLP-Spam-Classifier.git
cd NLP-Spam-Classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure Tesseract-OCR is installed and in your PATH**
- [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)

4. **Run the app**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:5000
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
