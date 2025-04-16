from flask import Flask, request, render_template
from preprocessing import preprocess_emails
import joblib
import pytesseract
from PIL import Image
import os

model = joblib.load('model_rf.pkl')

flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = 'uploads'

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get('email', '').strip()
    image = request.files.get('image')

    if not email_text and not image:
        return render_template("index.html", prediction_text="Please enter text or upload an image.")

    if image:
        image_path = os.path.join(flask_app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        extracted_text = pytesseract.image_to_string(Image.open(image_path))
        email_text = extracted_text.strip()
        os.remove(image_path)

    if not email_text.strip():
        return render_template("index.html", prediction_text="No valid text found from image or input.")

    prediction = model.predict([email_text])
    result = 'Spam 🛑' if prediction[0] == 1 else 'Not Spam ✅'

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)
    # flask_app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    flask_app.run(host="0.0.0.0", port=port)
