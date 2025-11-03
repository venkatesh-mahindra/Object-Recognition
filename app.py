import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 24px; font-weight: 600; color: #1f77b4;}
    .sub-header {font-size: 18px; font-weight: 500; color: #2c3e50;}
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        transition: all 0.3s ease;
    }
    
    /* Sliders */
    .stSlider>div>div>div>div {
        background-color: #1f77b4;
    }
    
    /* Select boxes */
    .stSelectbox>div>div>div {
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.title("🎛️ Controls")
    st.markdown("---")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)", "YOLOv8l (Large)"],
        index=0
    )
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the minimum confidence threshold for detections"
    )
    
    # Class selection
    selected_classes = st.multiselect(
        "Select Classes to Detect",
        ["person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic light", "stop sign"],
        default=["person", "car", "truck"]
    )
    
    st.markdown("---")
    st.markdown("### Source Selection")
    source_type = st.radio(
        "Select Input Source",
        ["Webcam", "Upload Video"],
        index=0
    )
    
    video_file = None
    if source_type == "Upload Video":
        video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    
    st.markdown("---")
    if st.button("🚀 Start Detection", use_container_width=True):
        st.session_state.run_detection = True
    
    if st.button("⏹️ Stop Detection", use_container_width=True):
        st.session_state.run_detection = False

# Main content
st.title("🔍 Real-Time Object Detection")
st.markdown("---")

# Initialize session state
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
    st.session_state.processed_video_path = None

# Load model with error handling for disk space
@st.cache_resource
def load_model(model_name):
    model_map = {
        "YOLOv8n (Nano)": "yolov8n.pt",
        "YOLOv8s (Small)": "yolov8s.pt",
        "YOLOv8m (Medium)": "yolov8m.pt",
        "YOLOv8l (Large)": "yolov8l.pt"
    }
    
    # Always use the nano model to save space
    model_name = "YOLOv8n (Nano)"
    st.warning(f"Using {model_name} to save disk space")
    
    try:
        return YOLO(model_map[model_name])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Please free up disk space or check your internet connection")
        st.stop()

try:
    model = load_model(model_type)
    class_names = model.names
    
    # Map class names to class IDs
    class_ids = [i for i, name in class_names.items() if name in selected_classes] if selected_classes else None
    
    # Display area
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    download_col1, download_col2 = st.columns([1, 1])
    
    if st.session_state.run_detection:
        if source_type == "Webcam" or video_file is not None:
            cap = None
            
            try:
                # Setup video writer for saving the processed video
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video.close()
                
                if source_type == "Webcam":
                    cap = cv2.VideoCapture(0)
                    # Test webcam
                    if not cap.isOpened():
                        st.error("❌ Could not access the webcam. Please ensure it's connected and not in use by another application.")
                        st.session_state.run_detection = False
                        st.stop()
                    else:
                        st.success("✅ Webcam connected successfully!")
                        # Get webcam resolution
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(video_file.read())
                    tfile.close()  # Close the file so it can be opened by OpenCV
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("❌ Could not open the video file. The file might be corrupted or in an unsupported format.")
                        st.session_state.run_detection = False
                        st.stop()
                    # Get video properties
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video.name, fourcc, 20.0, (frame_width, frame_height))
                
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.run_detection and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        if source_type == "Webcam":
                            st.error("⚠️ Lost connection to webcam. Please check if it's still connected.")
                        else:
                            st.warning("🎬 End of video file reached.")
                        st.session_state.run_detection = False
                        break
                    
                    # Run YOLO detection
                    results = model(frame, conf=confidence, classes=class_ids)
                    
                    # Create a copy of the frame for display
                    display_frame = frame.copy()
                    
                    # Draw detections
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            
                            # Draw rectangle and label (in BGR format for OpenCV)
                            color = (0, 165, 255)  # Orange in BGR
                            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            label = f"{class_names[cls]} {conf:.2f}"
                            cv2.putText(display_frame, label, (int(x1), int(y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    
                    # Display FPS
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Convert from BGR to RGB for display
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Write the processed frame to output video (in BGR format)
                    out.write(display_frame)
                    
                    # Display the frame
                    frame_placeholder.image(display_frame_rgb, use_container_width=True)
                    
                    # Update status
                    status_text.text(f"Detecting objects... FPS: {fps:.1f}")
                    
                    # Show download button only for uploaded videos (not webcam)
                    if source_type == "Upload Video" and not st.session_state.get('show_download_button', False):
                        st.session_state.processed_video_path = temp_video.name
                        st.session_state.show_download_button = True
                
                # Release everything when done
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                if 'out' in locals():
                    out.release()
                
                # Clean up temporary files
                try:
                    if 'tfile' in locals() and os.path.exists(tfile.name):
                        os.unlink(tfile.name)
                except:
                    pass
                
                # Show download button after processing is complete
                if source_type == "Upload Video":
                    with download_col1:
                        if st.download_button(
                            label="💾 Download Processed Video",
                            data=open(temp_video.name, 'rb').read(),
                            file_name="processed_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        ):
                            st.success("✅ Download started!")
                
                # Reset the flag for next run
                st.session_state.show_download_button = False
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.run_detection = False
            
            finally:
                # Release everything when done
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                if 'out' in locals():
                    out.release()
                
                # Save the path to the processed video
                st.session_state.processed_video_path = temp_video.name if 'temp_video' in locals() else None
        else:
            st.warning("Please upload a video file or select Webcam")
    else:
        # Show placeholder when not detecting
        if source_type == "Webcam":
            frame_placeholder.image("https://via.placeholder.com/800x450?text=Webcam+Feed+Will+Appear+Here", 
                                 use_container_width=True)
        else:
            frame_placeholder.image("https://via.placeholder.com/800x450?text=Upload+a+Video+or+Select+Webcam", 
                                 use_container_width=True)
        status_text.text("Ready to detect objects. Click 'Start Detection' to begin.")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Make sure you have an internet connection to download the model weights.")

# Footer
st.markdown("---")
st.markdown("### 📊 Detection Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Selected Classes", 
             ", ".join(selected_classes) if selected_classes else "All Classes")
with col2:
    st.metric("Model", model_type)
with col3:
    st.metric("Confidence Threshold", f"{confidence:.2f}")

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Add download button for processed video
if st.session_state.get('processed_video_path') and os.path.exists(st.session_state.processed_video_path):
    with open(st.session_state.processed_video_path, 'rb') as f:
        video_bytes = f.read()
    
    st.download_button(
        label="💾 Download Processed Video",
        data=video_bytes,
        file_name="processed_video.mp4",
        mime="video/mp4",
        key="download_button",
        help="Download the processed video with object detection"
    )

# Add a nice footer
st.markdown("""
---
### 🚀 About
This application demonstrates real-time object detection using YOLOv8 and Streamlit.
- **Model**: YOLOv8 (Ultralytics)
- **Backend**: Python, OpenCV
- **UI**: Streamlit

*Note: The first run may take some time to download the model weights.*
""")
