import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os

# Load the YOLOv8 model for trash overflow detection
model_1 = YOLO('last.pt')  # Trash overflow model

# Define class names for trash overflow model
class_names_1 = ['Broken trash can', 'Healthy trash can', 'Trash can closed', 'Trash can open', 'Trash over flow']

# Custom CSS for styling
st.markdown("""
<style>
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
This application detects <b>trash overflow</b>. Upload a video file to start detection. 🎥✨</p>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

# Start/Stop buttons
start_button = st.sidebar.button("▶️ Start Detection")
stop_button = st.sidebar.button("⏹️ Stop Detection")

# Placeholder for video display
video_placeholder = st.empty()

# Main function to process video frames
def process_video(file_path):
    global terminate_flag
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        st.error("❌ Error: Unable to open video file.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ End of video stream.")
            break

        # Run YOLOv8 inference for trash overflow detection
        results_1 = model_1(frame)

        # Visualize predictions on the frame for trash overflow model
        for result in results_1:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = f"{class_names_1[class_id]} ({conf:.2f})"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
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
    if uploaded_file is None:
        st.error("❌ Please upload a video file before starting detection.")
    else:
        terminate_flag = False
        # Save the uploaded file temporarily
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_video(file_path)

if stop_button:
    terminate_flag = True
    st.success("✅ Detection Stopped.")

# Additional content section
st.markdown("""
<h2 style='color: #d4af37;'>🌟 Why Choose Our System? 🌟</h2>
<ul>
    <li style='color: #ff6f61;'><b>Real-Time Detection:</b> Detects trash overflow in real-time using advanced AI models.</li>
    <li style='color: #2e7d32;'><b>User-Friendly Interface:</b> Easy-to-use interface with beautiful design and vibrant colors. 🎨</li>
    <li style='color: #d4af37;'><b>Sustainability:</b> Helps reduce environmental impact by promoting proper waste management. 🌍</li>
</ul>
""", unsafe_allow_html=True)