from flask import Flask, request, render_template, url_for
import numpy as np
import cv2
import joblib
import json
import pywt
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model and Class Dictionary
# -----------------------------
MODEL_PATH = "saved_model.pkl"
CLASS_DICT_PATH = "class_dictionary.json"

model = joblib.load(MODEL_PATH)
with open(CLASS_DICT_PATH, "r") as f:
    class_dict = json.load(f)

class_dict_rev = {v: k for k, v in class_dict.items()}

# -----------------------------
# Haarcascade detectors
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -----------------------------
# Helper Functions
# -----------------------------
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    return np.uint8(imArray_H)

def get_cropped_face_with_2_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # Draw rectangle for visualization
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            return roi_color, img
    return None, img

def prepare_image(img):
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))
    return combined_img.reshape(1, -1).astype(float)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cropped_img, annotated_img = get_cropped_face_with_2_eyes(img)
    if cropped_img is None:
        return render_template("index.html", error="No face with 2 eyes detected")

    processed_img = prepare_image(cropped_img)
    prediction = model.predict(processed_img)[0]
    prediction_proba = model.predict_proba(processed_img).max()

    # Save annotated image
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, annotated_img)

    return render_template(
        "index.html",
        prediction=class_dict_rev[prediction],
        confidence=round(float(prediction_proba) * 100, 2),
        image_url=url_for("static", filename=f"uploads/{filename}")
    )

if __name__ == "__main__":
    app.run(debug=True)
