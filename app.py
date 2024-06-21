from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import torch
from pathlib import Path
import os
import glob
import pathlib
import glob
import cv2
# Set pathlib settings untuk Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=Path('yolov5/best.pt').as_posix(), force_reload=True)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Define class names
class_names = ['early_blight', 'fresh', 'healthy', 'late_blight', 'leaf_mold', 'rotten', 'yellow_leaf_curl']

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform detection
            results = model(frame)
            
            # Extract bounding boxes and labels
            for *xyxy, conf, cls in results.xyxy[0].numpy():
                label = f'{class_names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run inference
            results = model(filepath)
            
            # Save results
            results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
            os.makedirs(results_dir, exist_ok=True)
            results.save(save_dir=results_dir)

            # Get the latest result directory
            actual_results_dir = max(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'results*')), key=os.path.getmtime)

            # Find the result image
            result_image_path = next((f for f in glob.glob(os.path.join(actual_results_dir, '*')) if f.endswith(('.jpg', '.jpeg', '.png'))), None)

            if result_image_path and os.path.exists(result_image_path):
                label = results.pandas().xyxy[0]['name'].iloc[0] if not results.pandas().xyxy[0].empty else 'No objects detected'
                result_image_url = url_for('uploaded_file', filename=os.path.relpath(result_image_path, start=app.config['UPLOAD_FOLDER']).replace("\\", "/"))
                return render_template('index.html', filename=result_image_url, label=label)
            else:
                return 'Error in processing the image.'

    return render_template('index.html')
    

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/detection', methods=['POST'])
def detection():
    return render_template('detection.html')

@app.route('/camera', methods=['POST'])
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
