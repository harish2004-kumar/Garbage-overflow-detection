import cv2
from ultralytics import YOLO
from flask import Flask, Response, render_template, request
import tempfile
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load the YOLOv8 models
model_1 = YOLO('last.pt')  # Trash overflow model
model_2 = YOLO('best (5).pt')  # Biodegradable vs non-biodegradable model

# Define class names for trash overflow model
class_names_1 = ['Broken trash can', 'Healthy trash can', 'Trash can closed', 'Trash can open', 'Trash over flow']

# Define class names for biodegradable vs non-biodegradable model
class_names_2 = ['biodegradable', 'non-biodegradable']

# Initialize counters
biodegradable_count = 0
non_biodegradable_count = 0

# Flag to indicate if the script should terminate
terminate_flag = False

# Function to send an email notification
def send_email_notification():
    sender_email = "officeprojects098@gmail.com"
    receiver_email = "chinna03022002@gmail.com"
    subject = "Trash Overflow Detected"
    body = "A trash overflow has been detected in the area. Please take necessary action."

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("officeprojects098@gmail.com", "jgeqrfaemnssxuki")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# Define a generator function to stream video frames to the web page
def generate(file_path):
    global biodegradable_count, non_biodegradable_count

    cap = cv2.VideoCapture(0) if file_path == "camera" else cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Unable to open video file or camera.")
        return b""

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame. Ending stream.")
            break

        # Run YOLOv8 inference on both models
        results_1 = model_1(frame)
        results_2 = model_2(frame)

        # Visualize predictions on the frame for trash overflow model
        for result in results_1:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_1[class_id]} ({conf:.2f})"

                if class_names_1[class_id] == 'Trash over flow' and conf > 0.5:
                    send_email_notification()

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 4
                text_color = (0, 165, 255)

                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size
                text_offset = 20
                text_bg_x1, text_bg_y1 = int(x1), int(y1) - text_height - text_offset
                text_bg_x2, text_bg_y2 = int(x1) + text_width + text_offset, int(y1)
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 165, 255), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(frame, label, (int(x1) + text_offset // 2, int(y1) - text_offset // 2),
                            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Visualize predictions on the frame for biodegradable vs non-biodegradable model
        for result in results_2:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_2[class_id]} ({conf:.2f})"

                if class_names_2[class_id] == 'biodegradable' and conf > 0.5:
                    biodegradable_count += 1
                    color = (0, 255, 0)
                elif class_names_2[class_id] == 'non-biodegradable' and conf > 0.5:
                    non_biodegradable_count += 1
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 3
                text_color = (255, 255, 255)

                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size
                text_offset = 20
                text_bg_x1, text_bg_y1 = int(x1), int(y1) - text_height - text_offset
                text_bg_x2, text_bg_y2 = int(x1) + text_width + text_offset, int(y1)
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame, label, (int(x1) + text_offset // 2, int(y1) - text_offset // 2),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Display the counters on the frame
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        font_thickness = 2
        text_color = (0, 255, 255)
        cv2.putText(frame, f"Biodegradable: {biodegradable_count}", (50, 50), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Non-Biodegradable: {non_biodegradable_count}", (50, 100), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        if terminate_flag:
            break

    cap.release()

# Flask application
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    return Response(generate(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag
    if request.method == 'POST':
        if request.form.get("camera") == "true":
            file_path = "camera"
        elif 'file' in request.files:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        else:
            file_path = None
        return render_template('index.html', file_path=file_path)
    else:
        terminate_flag = False
        return render_template('index.html')



@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    return "Process has been Terminated"

if __name__ == '__main__':
    app.run(debug=True)
