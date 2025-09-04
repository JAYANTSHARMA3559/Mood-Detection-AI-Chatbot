# Mood Detection & AI Chatbot

A real-time emotion detection and AI chatbot application that provides personalized responses based on a user's mood. The project features two distinct versions: a web-based application built with FastAPI and a desktop GUI application using Tkinter.

## 🚀 Key Features

* **Real-time Mood Detection:** Utilizes a pre-trained **Convolutional Neural Network (CNN)** to analyze facial expressions from a live webcam feed.
* **Intelligent Chatbot:** The chatbot provides dynamic and empathetic responses, offering support and encouragement that is tailored to the user's detected emotion.
* **Dual Interface Options:**
    * **Web Application:** A modern, accessible interface using **FastAPI** that runs on a web server.
    * **Desktop Application:** A self-contained graphical user interface (GUI) using **Tkinter**.
* **Asynchronous Communication:** The web application leverages **WebSockets** for efficient, low-latency communication between the server and the client, ensuring instant feedback.
* **Modular & Maintainable Code:** The project is designed with a clear, modular structure to separate concerns, making it easy to understand and expand.

## 💻 Technologies Used

* **Backend:** Python, FastAPI, Keras/TensorFlow
* **Computer Vision:** OpenCV, Haar Cascade
* **UI/Frontend:** HTML, CSS, JavaScript, WebSockets (for the web app), Tkinter (for the desktop app)
* **Development Tools:** Git

## 📂 Project Structure

```bash
.
├── static/                 # Static assets (CSS) for the web app
├── templates/              # HTML templates for the web app pages
├── app_copy.py             # The Tkinter desktop application code
├── emotiondetector.h5      # Pre-trained CNN model (excluded from Git)
├── README.md               # You are here!
└── requirements.txt        # Python dependencies
