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

# Flag to indicate if the script should terminate
terminate_flag = False

# Function to send an email notification
def send_email_notification():
    sender_email = "officeprojects098@gmail.com"  # Your email address
    receiver_email = "chinna03022002@gmail.com"  # Municipality's email address
    subject = "Trash Overflow Detected"
    body = "A trash overflow has been detected in the area. Please take necessary action."

    # Set up the server (Using Gmail as an example)
    server = smtplib.SMTP('smtp.gmail.com', 587)  # Change this for your email provider
    server.starttls()  # Enable security

    # Login to the email account
    server.login("officeprojects098@gmail.com", "jgeqrfaemnssxuki")  # Your email login credentials

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# Define a generator function to stream video frames to the web page
def generate(file_path):
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
                # Extract bounding box, confidence, and class ID for trash overflow model
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)  # Ensure class ID is an integer
                label = f"{class_names_1[class_id]} ({conf:.2f})"  # Class name and confidence

                # Check for trash overflow detection
                if class_names_1[class_id] == 'Trash over flow' and conf > 0.5:  # Adjust confidence threshold
                    send_email_notification()  # Send notification to municipality

                # Set font properties and sizes for trash overflow model
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2  # Increased font size
                font_thickness = 4  # Increased font thickness
                text_color = (0, 165, 255)  # Orange color (BGR format)

                # Get text size for background rectangle
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size
                text_offset = 20
                text_bg_x1, text_bg_y1 = int(x1), int(y1) - text_height - text_offset
                text_bg_x2, text_bg_y2 = int(x1) + text_width + text_offset, int(y1)
                # Draw the text background rectangle (semi-transparent)
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 165, 255), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Draw the bounding box for trash overflow model
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                # Put the text label for trash overflow model
                cv2.putText(frame, label, (int(x1) + text_offset // 2, int(y1) - text_offset // 2),
                            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Visualize predictions on the frame for biodegradable vs non-biodegradable model
        for result in results_2:
            for box in result.boxes.data:
                # Extract bounding box, confidence, and class ID for biodegradable vs non-biodegradable model
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)  # Ensure class ID is an integer
                label = f"{class_names_2[class_id]} ({conf:.2f})"  # Class name and confidence

                # Set color for bounding box and text
                if class_names_2[class_id] == 'biodegradable' and conf > 0.5:
                    color = (0, 255, 0)  # Green for biodegradable
                elif class_names_2[class_id] == 'non-biodegradable' and conf > 0.5:
                    color = (0, 0, 255)  # Red for non-biodegradable
                else:
                    color = (255, 255, 255)  # White for low confidence

                # Set font properties and sizes for biodegradable vs non-biodegradable model
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5  # Normal font size
                font_thickness = 3  # Normal font thickness
                text_color = (255, 255, 255)  # White color (BGR format)

                # Get text size for background rectangle
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size
                text_offset = 20
                text_bg_x1, text_bg_y1 = int(x1), int(y1) - text_height - text_offset
                text_bg_x2, text_bg_y2 = int(x1) + text_width + text_offset, int(y1)
                # Draw the text background rectangle (semi-transparent)
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Draw the bounding box for biodegradable vs non-biodegradable model
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

                # Put the text label for biodegradable vs non-biodegradable model
                cv2.putText(frame, label, (int(x1) + text_offset // 2, int(y1) - text_offset // 2),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break

        # Yield the JPEG data to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        # Exit the loop if the termination flag is set
        if terminate_flag:
            break

    cap.release()

# Flask application
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    return Response(generate(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route to serve the HTML page with the file upload form
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

# Route to stop video processing
@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    return "Process has been Terminated"

if __name__ == '__main__':
    app.run(debug=True)
