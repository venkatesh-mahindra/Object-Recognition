# 🚀 Real-Time Object Detection with Streamlit

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/YOLOv8-8.1.2-00BFFF.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-4.9.0.80-5C3EE8.svg" alt="OpenCV">
  <img src="https://img.shields.io/github/license/yourusername/object-detection-streamlit" alt="License">
</div>

A powerful and user-friendly web application for real-time object detection using YOLOv8 and Streamlit. This application allows you to detect objects in real-time using your webcam or from pre-recorded videos.

A powerful and user-friendly web application for real-time object detection using YOLOv8 and Streamlit. This application allows you to detect objects in real-time using your webcam or from pre-recorded videos.

## ✨ Features

- 🎥 **Webcam & Video Upload** - Detect objects live or from pre-recorded videos
- ⚡ **Real-time Processing** - Get instant detection results with FPS counter
- 🎯 **Customizable Detection** - Choose which objects to detect and set confidence thresholds
- 🎨 **Clean UI** - Modern and intuitive interface built with Streamlit
- 📊 **Performance Metrics** - View FPS and detection statistics in real-time

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-streamlit.git
   cd object-detection-streamlit
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser. If it doesn't, navigate to `http://localhost:8501`

3. Configure the detection settings in the sidebar:
   - Select the YOLO model (Nano, Small, Medium, or Large)
   - Adjust the confidence threshold
   - Choose which object classes to detect
   - Select input source (Webcam or Upload Video)

4. Click the "🚀 Start Detection" button to begin object detection

## 🌐 Supported Input Formats

- **Video Formats**: MP4, AVI, MOV, MKV
- **Webcam**: Any standard USB or built-in camera
- **Image Formats**: JPG, JPEG, PNG (static detection)

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" → "From existing repo"
4. Select your forked repository
5. Set the main file path to `app.py`
6. Click "Deploy!"

### Local Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-streamlit.git
   cd object-detection-streamlit
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Project Structure

```
object-detection-streamlit/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore file
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- **Video Formats**: MP4, AVI, MOV, MKV
- **Webcam**: Works with any standard USB webcam

## 📊 Models

The application supports multiple YOLOv8 models:

- **YOLOv8n (Nano)**: Fastest but less accurate
- **YOLOv8s (Small)**: Good balance of speed and accuracy
- **YOLOv8m (Medium)**: More accurate but slower
- **YOLOv8l (Large)**: Most accurate but slowest

## 📝 Notes

- The first run will download the YOLO model weights automatically (may take a few minutes)
- For best performance, use a GPU-enabled environment
- The application works on CPU but may experience reduced performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
