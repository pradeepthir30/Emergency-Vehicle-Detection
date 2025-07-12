# 🚨 Emergency Vehicle Detection Using IoT

This project uses a camera feed and YOLOv3 to detect emergency vehicles (like ambulances) and automatically control traffic lights using Arduino.

## 📌 Project Objective

To develop an intelligent traffic control system that detects the presence of emergency vehicles using deep learning and prioritizes their movement by controlling traffic signals through Arduino.

---

## 📁 Files in This Repository

| File Name              | Description                                    |
|------------------------|------------------------------------------------|
| `video_web.py`         | Python script for running YOLO detection       |
| `yolov3_custom.cfg`    | YOLOv3 configuration file (custom-trained)     |
| `classes.names`        | List of object class labels (e.g., ambulance)  |
| `cars.xml`             | Haar cascade XML file for car detection (optional) |

---

## 🧠 Technologies Used

- Python 3.x
- OpenCV
- YOLOv3 (Darknet version)
- Arduino Uno
- Serial Communication (`pyserial`)
- Google Drive (for hosting `.weights`)

---

## 🧰 Requirements

Install the required Python libraries:

```bash
pip install opencv-python numpy pyserial

**▶️ How to Run the Detection**
Download yolov3_custom.weights

Place the weights file in the same folder as video_web.py

Run the detection script:

python video_web.py

If an emergency vehicle is detected in the video feed, a signal is sent to the Arduino via serial communication to change the traffic light to green.

**🎬 Demo**
👉 https://drive.google.com/file/d/11mIC3iCosgs9ccggwLbivFQX_5grV4fB/view?usp=sharing

**👨‍💻 Team**
Pradeepthi Rajuladevi

Final Year B.Tech Project Team

**📄 License**
This project is for academic and educational purposes only.

**yolov3_custom.weights:**
[https://your-google-drive-link-here](https://drive.google.com/file/d/16sjxonykTDxnP8Pvs5Sn2A1DxHed8STW/view?usp=sharing)
