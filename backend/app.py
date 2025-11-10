from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2
from deepface import DeepFace
import numpy as np
import json
import os
from datetime import datetime
import csv
import mediapipe as mp
import tensorflow as tf

# CPU optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

app = Flask(__name__)

FACES_DIR = 'faces'
USERS_FILE = 'data/users.json'
ATTENDANCE_FILE = 'data/attendance.csv'
SLEEPING_DIR = 'data/sleeping'
SLEEPING_LOG_FILE = 'data/sleeping_log.csv'

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs(SLEEPING_DIR, exist_ok=True)

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def mark_attendance(name, roll_number, date=None):
    today = date if date else datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')
    
    file_exists = os.path.exists(ATTENDANCE_FILE)
    
    if file_exists:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3 and row[1] == roll_number and row[2] == today:
                    return False
    
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Roll Number', 'Date', 'Time'])
        writer.writerow([name, roll_number, today, time])
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/api/check_face', methods=['POST'])
def api_check_face():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_data = image_data.split(',')[1]
    import base64
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    try:
        faces = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=True)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
    except:
        return jsonify({'success': False, 'message': 'No face detected'})
    
    temp_path = 'temp_check.jpg'
    cv2.imwrite(temp_path, img)
    
    users = load_users()
    if not users:
        os.remove(temp_path)
        return jsonify({'success': True, 'message': 'Face is unique'})
    
    for user in users:
        try:
            result = DeepFace.verify(temp_path, user['image_path'], 
                                    detector_backend='opencv', 
                                    model_name='Facenet',
                                    enforce_detection=False,
                                    distance_metric='cosine',
                                    align=False)
            if result['verified'] and result['distance'] < 0.4:
                os.remove(temp_path)
                return jsonify({'success': False, 'message': f'Face already registered as {user["name"]} ({user["roll_number"]})'}) 
        except:
            continue
    
    os.remove(temp_path)
    return jsonify({'success': True, 'message': 'Face is unique'})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get('name')
    roll_number = data.get('roll_number')
    gender = data.get('gender')
    image_data = data.get('image')
    
    if not name or not roll_number or not gender or not image_data:
        return jsonify({'success': False, 'message': 'Missing data'})
    
    users = load_users()
    if any(u['roll_number'] == roll_number for u in users):
        return jsonify({'success': False, 'message': 'Roll number already exists'})
    
    image_data = image_data.split(',')[1]
    import base64
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    image_path = os.path.join(FACES_DIR, f'{roll_number}.jpg')
    cv2.imwrite(image_path, img)
    
    users.append({
        'name': name,
        'roll_number': roll_number,
        'gender': gender,
        'image_path': image_path
    })
    save_users(users)
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.json
    image_data = data.get('image')
    custom_date = data.get('date')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_data = image_data.split(',')[1]
    import base64
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    temp_path = 'temp_capture.jpg'
    cv2.imwrite(temp_path, img)
    
    users = load_users()
    for user in users:
        try:
            result = DeepFace.verify(temp_path, user['image_path'], 
                                    detector_backend='opencv', 
                                    model_name='Facenet',
                                    enforce_detection=False,
                                    distance_metric='cosine',
                                    align=False)
            
            if result['verified'] and result['distance'] < 0.4:
                os.remove(temp_path)
                marked = mark_attendance(user['name'], user['roll_number'], custom_date)
                if marked:
                    return jsonify({
                        'success': True,
                        'name': user['name'],
                        'roll_number': user['roll_number'],
                        'message': 'Attendance marked successfully'
                    })
                else:
                    return jsonify({
                        'success': True,
                        'name': user['name'],
                        'roll_number': user['roll_number'],
                        'message': 'Attendance already marked today'
                    })
        except:
            continue
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return jsonify({'success': False, 'message': 'Face not recognized'})

@app.route('/api/users')
def api_users():
    return jsonify(load_users())

@app.route('/api/update/<roll_number>', methods=['PUT'])
def api_update(roll_number):
    data = request.json
    users = load_users()
    user = next((u for u in users if u['roll_number'] == roll_number), None)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    user['name'] = data.get('name', user['name'])
    user['gender'] = data.get('gender', user.get('gender'))
    
    if 'image' in data:
        image_data = data['image'].split(',')[1]
        import base64
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        try:
            faces = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=True)
            if len(faces) == 0:
                return jsonify({'success': False, 'message': 'No face detected in image'})
        except:
            return jsonify({'success': False, 'message': 'No face detected in image'})
        
        image_path = os.path.join(FACES_DIR, f'{roll_number}.jpg')
        cv2.imwrite(image_path, img)
        global face_encodings_cache
        face_encodings_cache = {}
    
    save_users(users)
    
    return jsonify({'success': True, 'message': 'Student updated successfully'})

@app.route('/api/delete/<roll_number>', methods=['DELETE'])
def api_delete(roll_number):
    users = load_users()
    user = next((u for u in users if u['roll_number'] == roll_number), None)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    if os.path.exists(user['image_path']):
        os.remove(user['image_path'])
    
    users = [u for u in users if u['roll_number'] != roll_number]
    save_users(users)
    
    return jsonify({'success': True, 'message': 'User deleted successfully'})

@app.route('/students')
def students():
    return render_template('students.html')

@app.route('/surveillance')
def surveillance():
    return render_template('surveillance.html')

@app.route('/faces/<filename>')
def serve_face(filename):
    return send_from_directory(FACES_DIR, filename)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.3, min_tracking_confidence=0.3, refine_landmarks=False)
face_encodings_cache = {}

def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def load_face_encodings():
    global face_encodings_cache
    if face_encodings_cache:
        return face_encodings_cache
    
    users = load_users()
    for user in users:
        try:
            img = cv2.imread(user['image_path'])
            result = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
            face_encodings_cache[user['roll_number']] = {
                'name': user['name'],
                'encoding': result[0]['embedding']
            }
        except:
            continue
    return face_encodings_cache

def log_sleeping(name, roll_number, image_path):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.exists(SLEEPING_LOG_FILE)
    with open(SLEEPING_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Roll Number', 'Date', 'Time', 'Image'])
        writer.writerow([name, roll_number, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), image_path])

@app.route('/api/sleeping_logs')
def api_sleeping_logs():
    logs = []
    if os.path.exists(SLEEPING_LOG_FILE):
        with open(SLEEPING_LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            logs = list(reader)
    return jsonify(logs)

@app.route('/api/detect_faces', methods=['POST'])
def api_detect_faces():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'faces': []})
    
    import base64
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result_faces = []
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        h, w, _ = img.shape
        
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            
            left_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in left_eye])
            right_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in right_eye])
            
            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            avg_ear = (left_ear + right_ear) / 2.0
            eyes_open = 2 if avg_ear > 0.2 else 0
            
            xs = [face_landmarks.landmark[i].x * w for i in range(468)]
            ys = [face_landmarks.landmark[i].y * h for i in range(468)]
            x, y = int(min(xs)), int(min(ys))
            fw, fh = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            
            result_faces.append({
                'x': x,
                'y': y,
                'w': fw,
                'h': fh,
                'eyes': eyes_open
            })
    
    return jsonify({'success': True, 'faces': result_faces})

@app.route('/api/save_sleeping_image', methods=['POST'])
def api_save_sleeping_image():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False})
    
    import base64
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sleeping_image_path = os.path.join(SLEEPING_DIR, f"sleeping_{timestamp}.jpg")
    cv2.imwrite(sleeping_image_path, img)
    
    return jsonify({'success': True, 'image_path': sleeping_image_path})

@app.route('/api/match_sleeping_face', methods=['POST'])
def api_match_sleeping_face():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'success': False, 'message': 'Image not found'})
    
    users = load_users()
    matched_user = None
    
    for user in users:
        try:
            result = DeepFace.verify(image_path, user['image_path'], 
                                    detector_backend='opencv', 
                                    model_name='Facenet',
                                    enforce_detection=False,
                                    distance_metric='cosine',
                                    align=False)
            if result['verified'] and result['distance'] < 0.4:
                matched_user = user
                break
        except:
            continue
    
    if matched_user:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_path = os.path.join(SLEEPING_DIR, f"{matched_user['roll_number']}_{timestamp}.jpg")
        os.rename(image_path, new_path)
        log_sleeping(matched_user['name'], matched_user['roll_number'], new_path)
        return jsonify({'success': True, 'name': matched_user['name'], 'roll_number': matched_user['roll_number']})
    
    return jsonify({'success': False, 'message': 'Face not recognized'})

@app.route('/data/sleeping/<filename>')
def serve_sleeping(filename):
    return send_from_directory(SLEEPING_DIR, filename)

@app.route('/api/clear_database', methods=['POST'])
def api_clear_database():
    try:
        import shutil
        if os.path.exists(USERS_FILE):
            os.remove(USERS_FILE)
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)
        if os.path.exists(SLEEPING_LOG_FILE):
            os.remove(SLEEPING_LOG_FILE)
        if os.path.exists(FACES_DIR):
            shutil.rmtree(FACES_DIR)
            os.makedirs(FACES_DIR)
        if os.path.exists(SLEEPING_DIR):
            shutil.rmtree(SLEEPING_DIR)
            os.makedirs(SLEEPING_DIR)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
