from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from deepface import DeepFace
import cv2
import numpy as np
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Preload models by analyzing a placeholder image
def preload_models():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder black image for caching models
    DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

# Call preload_models once when the app starts
preload_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_faces_and_mood(image_path):
    img = cv2.imread(image_path)

    # Validate that the image loaded successfully
    if img is None:
        print(f"Could not read image at path: {image_path}")
        return None, []

    try:
        # Analyze the image
        analysis_results = DeepFace.analyze(
            img,
            actions=['emotion', 'age', 'gender', 'race'],
            enforce_detection=False
        )
    except Exception as e:
        print("Error in face and mood detection:", e)
        return None, []

    mood_data = []
    if isinstance(analysis_results, dict):
        analysis_results = [analysis_results]

    for face_analysis in analysis_results:
        x, y, w, h = face_analysis['region']['x'], face_analysis['region']['y'], face_analysis['region']['w'], face_analysis['region']['h']
        mood = face_analysis['dominant_emotion']
        age = face_analysis['age']
        gender = face_analysis['gender']
        race = face_analysis['dominant_race']
        # emotion_confidences = face_analysis['emotion']

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        mood_data.append({
            'coordinates': (x, y, w, h),
            'mood': mood,
            'age': age,
            'gender': gender,
            'race': race,
            # 'emotion_confidences': emotion_confidences
        })

    # Generate a unique filename for the result image
    result_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    return result_filename, mood_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image_filename, mood_data = detect_faces_and_mood(file_path)
        return render_template('result.html', image_filename=image_filename, mood_data=mood_data)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)