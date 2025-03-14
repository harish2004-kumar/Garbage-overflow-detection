import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load the YOLOv8 models
model_1 = YOLO('last.pt')  # Trash overflow model
model_2 = YOLO('best (5).pt')  # Biodegradable vs non-biodegradable model
model_3 = YOLO('best6.pt')  # New model for detecting 42 classes of waste materials

# Define class names for trash overflow model
class_names_1 = ['Broken trash can', 'Healthy trash can', 'Trash can closed', 'Trash can open', 'Trash over flow']

# Define class names for biodegradable vs non-biodegradable model
class_names_2 = ['biodegradable', 'non-biodegradable']

# Define class names for the new model
class_names_3 = [
    'Aerosols', 'Aluminum can', 'Aluminum caps', 'Cardboard', 'Cellulose', 'Ceramic', 'Combined plastic',
    'Container for household chemicals', 'Disposable tableware', 'Electronics', 'Foil', 'Furniture', 'Glass bottle',
    'Iron utensils', 'Liquid', 'Metal shavings', 'Milk bottle', 'Organic', 'Paper bag', 'Paper cups', 'Paper shavings',
    'Paper', 'Papier mache', 'Plastic bag', 'Plastic bottle', 'Plastic can', 'Plastic canister', 'Plastic caps',
    'Plastic cup', 'Plastic shaker', 'Plastic shavings', 'Plastic toys', 'Postal packaging', 'Printing industry',
    'Scrap metal', 'Stretch film', 'Tetra pack', 'Textile', 'Tin', 'Unknown plastic', 'Wood', 'Zip plastic bag'
]

# Function to send an email notification
def send_email_notification():
    sender_email = "wellm1696@gmail.com"  # Your email address
    receiver_email = "harish192415@gmail.com"  # Municipality's email address
    subject = "🚨 Trash Overflow Detected 🚨"
    body = "A trash overflow has been detected in the area. Please take necessary action immediately! 📬"

    # Set up the server (Using Gmail as an example)
    server = smtplib.SMTP('smtp.gmail.com', 587)  # Change this for your email provider
    server.starttls()  # Enable security
    server.login("wellm1696@gmail.com", "xafqgimbwbfxfoqd")  # Your email login credentials

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# Custom CSS for styling
st.markdown("""
<style>
/* Apply custom font */
body {
    font-family: 'Times New Roman', serif;
}
h1 {
    color: #ff6f61; /* Coral red title */
    text-align: center; /* Center align */
    font-size: 48px; /* Larger font size */
    margin-bottom: 20px; /* Spacing below */
}
p {
    color: #2e7d32; /* Dark green text */
    font-size: 18px; /* Larger font size */
    line-height: 1.6; /* Spacing between lines */
}
.stButton>button {
    background-color: #ffab40; /* Orange background */
    color: white; /* White text */
    border-radius: 10px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 16px; /* Font size */
    border: none; /* No border */
    cursor: pointer; /* Pointer on hover */
}
.stButton>button:hover {
    background-color: #ff8f00; /* Darker orange on hover */
}
.sidebar .sidebar-content {
    background-color: #fff3e0; /* Light orange background for sidebar */
    padding: 20px; /* Padding */
    border-radius: 10px; /* Rounded corners */
}
</style>
""", unsafe_allow_html=True)

# Streamlit app title and description
st.title("🌍 Smart Waste Management System 🌍")
st.markdown(f"""
<p>Welcome to the <b>Smart Waste Management System</b>, where technology meets sustainability! 🌱✨ 
This application detects <b>trash overflow</b>, classifies waste as <b>biodegradable</b> or <b>non-biodegradable</b>, 
and identifies <b>42 specific types of waste materials</b>. Upload a video file or use your webcam to start detection. 🎥✨</p>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("⚙️ Settings")
source_type = st.sidebar.radio("📂 Select Input Source", ["🎥 Webcam", "📤 Upload Video"])
terminate_flag = False

# File uploader for video input
if source_type == "📤 Upload Video":
    uploaded_file = st.sidebar.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    file_path = "camera"

# Start/Stop buttons
start_button = st.sidebar.button("▶️ Start Detection")
stop_button = st.sidebar.button("⏹️ Stop Detection")

# Placeholder for video display
video_placeholder = st.empty()

# Main function to process video frames
def process_video(file_path):
    global terminate_flag
    cap = cv2.VideoCapture(0) if file_path == "camera" else cv2.VideoCapture(file_path)
    if not cap.isOpened():
        st.error("❌ Error: Unable to open video file or camera.")
        return
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ End of video stream.")
            break

        # Run YOLOv8 inference on all three models
        results_1 = model_1(frame)
        results_2 = model_2(frame)
        results_3 = model_3(frame)

        # Visualize predictions on the frame for trash overflow model
        for result in results_1:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_1[class_id]} ({conf:.2f})"
                if class_names_1[class_id] == 'Trash over flow' and conf > 0.5:
                    send_email_notification()
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualize predictions on the frame for biodegradable vs non-biodegradable model
        for result in results_2:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_2[class_id]} ({conf:.2f})"
                color = (0, 255, 0) if class_names_2[class_id] == 'biodegradable' else (0, 0, 255)
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualize predictions on the frame for the new model (42 classes)
        for result in results_3:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_3[class_id]} ({conf:.2f})"
                color = (255, 0, 255)  # Magenta for the new model
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB")

        # Exit the loop if the termination flag is set
        if terminate_flag:
            break
    cap.release()

# Start/Stop logic
if start_button:
    terminate_flag = False
    process_video(file_path)

if stop_button:
    terminate_flag = True
    st.success("✅ Detection Stopped.")

# Additional content section
st.markdown("""
<h2 style='color: #d4af37;'>🌟 Why Choose Our System? 🌟</h2>
<ul>
    <li style='color: #ff6f61;'><b>Real-Time Detection:</b> Detects trash overflow and waste types in real-time using advanced AI models.</li>
    <li style='color: #2e7d32;'><b>Email Alerts:</b> Get instant notifications when trash overflow is detected. 📧</li>
    <li style='color: #ffab40;'><b>User-Friendly Interface:</b> Easy-to-use interface with beautiful design and vibrant colors. 🎨</li>
    <li style='color: #d4af37;'><b>Sustainability:</b> Helps reduce environmental impact by promoting proper waste management. 🌍</li>
</ul>
""", unsafe_allow_html=True)