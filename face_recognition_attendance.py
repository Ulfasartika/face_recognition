import os
import cv2
import time
import numpy as np
import tensorflow as tf
import mysql.connector
from datetime import datetime
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from mtcnn import MTCNN
from inception_resnet_v1 import InceptionResNetV1

app = Flask(__name__)
# Izinkan CORS untuk semua endpoint API, termasuk video_feed
CORS(app, resources={r"/api/*": {"origins": ["http://127.0.0.1:8000", "http://localhost:8000"]}}, supports_credentials=True)

# --- MySQL Config ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'fazzin'
}

# --- Path and config ---
DATA_DIR = 'face_data'
ATTENDANCE_LOG = 'attendance_log.txt'
THRESHOLD = 0.8
UPDATE_THRESHOLD = 0.6  # Threshold untuk verifikasi wajah saat update (lebih ketat dari pengenalan)

# --- Ensure directories exist ---
os.makedirs(DATA_DIR, exist_ok=True)

# --- Load model ---
facenet_model = InceptionResNetV1()
facenet_model.trainable = False

# --- MTCNN detector ---
detector = MTCNN()

# --- MySQL Connection ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise

# --- Utilities ---
def save_embedding(user_id, embedding):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        embedding_bytes = embedding.tobytes()
        query = "INSERT INTO face_embeddings (user_id, embedding) VALUES (%s, %s)"
        cursor.execute(query, (user_id, embedding_bytes))
        conn.commit()
        
        # Batasi maksimum 5 embedding per user
        cursor.execute("SELECT id FROM face_embeddings WHERE user_id = %s ORDER BY created_at DESC LIMIT 5, 1")
        old_embedding = cursor.fetchone()
        if old_embedding:
            cursor.execute("DELETE FROM face_embeddings WHERE id = %s", (old_embedding[0],))
            conn.commit()
        
        print(f"[INFO] Saved embedding for user {user_id}")
    except mysql.connector.Error as e:
        print(f"[ERROR] Failed to save embedding: {e}")
    finally:
        cursor.close()
        conn.close()

def load_embeddings():
    embeddings = {}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, embedding FROM face_embeddings")
        rows = cursor.fetchall()
        
        for user_id, emb_bytes in rows:
            embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.setdefault(str(user_id), []).append(embedding)
        
        print(f"[INFO] Loaded embeddings for {len(embeddings)} user(s)")
    except mysql.connector.Error as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
    finally:
        cursor.close()
        conn.close()
    return embeddings

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

def get_embedding(face_image):
    preprocessed = preprocess_image(face_image)
    embedding = facenet_model(preprocessed, training=False).numpy()[0]
    return embedding

# --- API Endpoints ---
@app.route('/api/register', methods=['POST'])
def register():
    user_id = request.form.get('user_id')
    image_data = request.files.get('image')

    if not user_id or not image_data:
        return jsonify({"status": "error", "message": "User ID and image are required"}), 400

    # Decode image
    npimg = np.frombuffer(image_data.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect face
    faces = detector.detect_faces(frame)
    if not faces:
        return jsonify({"status": "error", "message": "Face not detected. Please ensure your face is clearly visible and unobstructed."}), 400

    # Save embedding
    x, y, w, h = faces[0]['box']
    face_image = frame[y:y+h, x:x+w]
    embedding = get_embedding(face_image)
    save_embedding(user_id, embedding)

    # Save image for debugging
    filename = os.path.join(DATA_DIR, f"{user_id}_{int(time.time())}.jpg")
    cv2.imwrite(filename, face_image)

    return jsonify({"status": "success", "message": f"Face registered for {user_id}"}), 200

@app.route('/api/update_face', methods=['POST'])
def update_face():
    user_id = request.form.get('user_id')
    image_data = request.files.get('image')

    if not user_id or not image_data:
        return jsonify({"status": "error", "message": "User ID and image are required"}), 400

    # Decode image
    npimg = np.frombuffer(image_data.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect face
    faces = detector.detect_faces(frame)
    if not faces:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    # Get new face embedding
    x, y, w, h = faces[0]['box']
    face_image = frame[y:y+h, x:x+w]
    new_embedding = get_embedding(face_image)

    # Load existing embeddings for this user
    embeddings = load_embeddings()
    if str(user_id) not in embeddings:
        return jsonify({"status": "error", "message": "No previous face data found for this user"}), 400

    # Verify new face against existing embeddings
    min_dist = float('inf')
    for emb in embeddings[str(user_id)]:
        dist = np.linalg.norm(new_embedding - emb)
        if dist < min_dist:
            min_dist = dist

    if min_dist > UPDATE_THRESHOLD:
        return jsonify({
            "status": "error",
            "message": "New face does not match previous face data",
            "distance": float(min_dist)
        }), 400

    # Save new embedding if verification passes
    save_embedding(user_id, new_embedding)

    # Save image for debugging
    filename = os.path.join(DATA_DIR, f"{user_id}_update_{int(time.time())}.jpg")
    cv2.imwrite(filename, face_image)

    return jsonify({"status": "success", "message": f"Face updated for user {user_id}"}), 200

@app.route('/api/recognize', methods=['POST'])
def recognize():
    image_data = request.files.get('image')
    if not image_data:
        return jsonify({"status": "error", "message": "Image is required"}), 400

    # Decode image
    npimg = np.frombuffer(image_data.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect face
    faces = detector.detect_faces(frame)
    if not faces:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    # Recognize face
    x, y, w, h = faces[0]['box']
    face_img = frame[y:y+h, x:x+w]
    test_emb = get_embedding(face_img)

    embeddings = load_embeddings()
    best_match = "Unknown"
    min_dist = THRESHOLD

    for known_id, emb_list in embeddings.items():
        for emb in emb_list:
            dist = np.linalg.norm(test_emb - emb)
            if dist < min_dist:
                min_dist = dist
                best_match = known_id

    if best_match != "Unknown":
        with open(ATTENDANCE_LOG, 'a') as f:
            f.write(f"{best_match},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        return jsonify({
            "status": "success",
            "user_id": best_match,
            "message": f"Recognized {best_match}",
            "distance": float(min_dist)
        }), 200
    else:
        return jsonify({"status": "error", "message": "Face not recognized", "distance": float(min_dist)}), 400

@app.route('/api/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        cap.release()
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8000'
    return response

# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)