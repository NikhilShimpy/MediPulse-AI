
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask setup
app = Flask(__name__)
app.secret_key = 'healthsync_secret_key'

# Fix: absolute database path
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'database', 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
MODEL_PATH = os.path.join(basedir, 'model', 'acne_classifier.h5')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(os.path.join(basedir, 'database'), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(basedir, 'model'), exist_ok=True)

# Load ML model
model = load_model(MODEL_PATH)
class_labels = ['Mild Acne', 'Moderate Acne', 'Severe Acne']

# DB setup
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        if User.query.filter_by(email=email).first():
            return 'User already exists'
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user'] = user.email
            return redirect(url_for('dashboard'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/emergency')
def emergency():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('emergency.html')

@app.route('/medicalquery')
def medical_query():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('medicalquery.html', user=session['user'])

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/generalquery')
def general_query():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('generalquery.html', user=session['user'])


@app.route('/book-appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        date = request.form['date']
        time = request.form['time']
        doctor = request.form['doctor']
        reason = request.form['reason']

        # Optional: Save to database, send email, or log
        print(f"Appointment booked by {fullname} ({email}) with {doctor} on {date} at {time}. Reason: {reason}")
        
        return render_template('appointment_success.html', name=fullname)

    return render_template('Bookappointment.html', user=session['user'])


@app.route('/get-hospitals')
def get_hospitals():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    radius_km = float(request.args.get('radius', 5))
    radius_m = int(radius_km * 1000)

    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        res = requests.post("http://overpass-api.de/api/interpreter", data=query, timeout=10)
        elements = res.json().get('elements', [])
    except Exception as e:
        return jsonify([])

    hospitals = []
    for e in elements:
        name = e['tags'].get('name', 'Unnamed Hospital')
        latlon = (e.get('lat'), e.get('lon')) if 'lat' in e else (e['center']['lat'], e['center']['lon']) if 'center' in e else None
        if latlon:
            hospitals.append({'name': name, 'lat': latlon[0], 'lon': latlon[1]})
    return jsonify(hospitals)

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if 'user' not in session:
        return redirect(url_for('login'))

    diagnosis = None
    suggestions = []
    uploaded_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image = filename

            # ML prediction
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            class_index = np.argmax(preds)
            diagnosis = class_labels[class_index]

            # Suggestions
            if diagnosis == 'Mild Acne':
                suggestions = [
                    "Use a gentle cleanser twice daily",
                    "Avoid scrubbing or over-washing",
                    "Stay hydrated"
                ]
            elif diagnosis == 'Moderate Acne':
                suggestions = [
                    "Use benzoyl peroxide or salicylic acid",
                    "Avoid touching or popping pimples",
                    "Consider visiting a dermatologist"
                ]
            elif diagnosis == 'Severe Acne':
                suggestions = [
                    "Consult a dermatologist immediately",
                    "Oral antibiotics or isotretinoin may be recommended",
                    "Maintain a consistent skincare routine"
                ]

    return render_template('disease_detection.html',
                           uploaded_image=uploaded_image,
                           diagnosis=diagnosis,
                           suggestions=suggestions)

# Start app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
