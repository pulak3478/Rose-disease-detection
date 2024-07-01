from flask import Flask
import os
from flask import Flask, render_template, request, redirect, url_for,jsonify, Response 
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
import numpy as np
from pyngrok import ngrok
import sqlite3
import uuid
import time
import threading
from PIL import Image
ngrok.set_auth_token("2fBP22Oq2BR4yn9KUkznDCtvLUM_28AczNttJLFSNUrGvfN2h")
public_url=ngrok.connect(5000).public_url
port_no=5000

app = Flask(__name__)
ngrok.set_auth_token("2fBP22Oq2BR4yn9KUkznDCtvLUM_28AczNttJLFSNUrGvfN2h")
model = YOLO('./runs/classify/train/last.pt')
models = YOLO('last.pt')

DATABASE = 'button_counts.db'
def create_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS button_counts
                 (id INTEGER PRIMARY KEY, count INTEGER)''')
    conn.commit()
    conn.close()
create_db()

def predict_disease(image_stream):
    
    image_array = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform prediction on the image
    results = model(image_array, show=True)

    # Extract class names from the model
    names_dict = results[0].names

    # Extract relevant information from the prediction
    probs = results[0].probs.data.tolist()
    prediction = names_dict[probs.index(max(probs))]
    return prediction,probs, names_dict

@app.route('/')
def home():
   
    image_exists = os.path.exists('static/image/temp.JPG')
    if image_exists:
        image_url = f'/static/image/temp.JPG'
    else:
        image_url = None

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT SUM(count) FROM button_counts")
    total_count = c.fetchone()[0]
    conn.close()
    return render_template('index.html',total_count=total_count, prediction=None, probs=None, names_dict=None, image_exists=image_exists, image_url=image_url,background_image_url="/static/pf.jpg")

@app.route('/increment', methods=['POST'])
def increment_count():
    # Increment count in the database
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO button_counts (count) VALUES (1)")
    conn.commit()
    conn.close()
    return redirect(url_for('home'))

@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        # Increment count in the database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO button_counts (count) VALUES (1)")
        conn.commit()
        conn.close()
        return redirect(url_for('about'))
    else:
        # Fetch total count from the database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT SUM(count) FROM button_counts")
        total_count = c.fetchone()[0]
        conn.close()
        return render_template('about.html', total_count=total_count)
   
@app.route('/page', methods=['GET', 'POST'])
def page():
    if request.method == 'GET':
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT SUM(count) FROM button_counts")
        total_count = c.fetchone()[0]
        conn.close()
        return render_template('page.html', total_count=total_count)
   
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    image_stream = BytesIO()
    file.save(image_stream)
    image_stream.seek(0)

    # Pass the image stream to the YOLOv5 model for prediction
    prediction, probs, names_dict = predict_disease(image_stream)

    image_exists = prediction is not None

    # Encode the image data as base64 and create data URI
    if image_exists:
        image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        image_url = f'data:image/jpeg;base64,{image_data}'
    else:
        image_url = None

    # Render the template with the prediction result
    return render_template('index.html', prediction=prediction, probs=probs, names_dict=names_dict, image_exists=image_exists, image_url=image_url,background_image_url="/static/pf.jpg")
@app.route('/pure',methods=['GET','POST'])
def pure():
    image_path = os.path.exists('static/image/temp.JPG')
    if image_path:
        image_url = f'/static/image/temp.JPG'
    else:
        image_url = None
    if request.method == 'POST':
        # Increment count in the database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO button_counts (count) VALUES (1)")
        conn.commit()
        conn.close()
        return redirect(url_for('pure'))
    else:
        # Fetch total count from the database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT SUM(count) FROM button_counts")
        total_count = c.fetchone()[0]
        conn.close()
       
        return render_template('pure.html', total_count=total_count,image_path=image_path,image_url=image_url,background_image_url="/static/pf.jpg")

def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

@app.route('/predictions', methods=["GET","POST"])
def predictions():

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            unique_filename = generate_unique_filename(file.filename)
            image_path = os.path.join('static', 'images', unique_filename)
            file.save(image_path)
            # Run inference on the uploaded image
            results = models(image_path)  # results list

            # Visualize the results
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

                # Save the result image
                result_image_path = os.path.join('static','images',  unique_filename)
                im_rgb.save(result_image_path)

            # Render the HTML template with the result image path
            return render_template('pure.html', image_url=result_image_path, image_path=image_path,background_image_url="/static/pf.jpg")
    
@app.route('/live_feed_page')
def live_feed_page():
    return render_template('live_feed.html')

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    while True:
        success, frame = cap.read()

        if success:
            # Perform prediction on the frame using your YOLO model
            results = models(frame)
            annotated_frame = results[0].plot()

            # Convert the annotated frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame bytes as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

@app.route('/vidpred', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            unique_filename = generate_unique_filename(file.filename)
            video_path = os.path.join('static', 'images',unique_filename)
            file.save(video_path)
            
            return redirect(url_for('video_feed', video_path=video_path))
    
    return render_template('pure.html')

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = models(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()
    os.remove(video_path)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', None)
    
    if video_path:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return 'Error: No video file provided.'

def delete_images_after_delay():
    while True:
        time.sleep(86400)  # Wait 1 day
        image_folder = 'static/images'
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Flask route to delete images after 2 minutes
@app.route('/delete', methods=['GET'])
def delete():
    threading.Thread(target=delete_images_after_delay).start()
    return jsonify({"message": "Images will be deleted continuously after 2 minutes."})


print(f"To acces the Gloable link please click\n{public_url}")
if __name__ == '__main__':
    app.run(port=5000)
