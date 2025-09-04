# Mood Detection & AI Chatbot

A real-time, empathetic chatbot that uses computer vision to detect and respond to a user's emotional state. This project features two distinct applications: a web-based interface and a desktop GUI.

## Problem Statement
Human-computer interaction is becoming more sophisticated, requiring systems to be contextually aware and emotionally intelligent. This project addresses the need for an interactive agent that can understand a user's mood through facial expressions. The goal is to provide supportive, personalized responses, fostering a more natural and engaging user experience.

## Dataset
* **Model:** Pre-trained Convolutional Neural Network (CNN) model (`emotiondetector.h5`) for facial emotion recognition.
* **Source:** Trained on a public dataset of facial expressions (e.g., FER-2013).

## Methods & Tools
* **AI Model (Keras/TensorFlow)**
    * A CNN model is used for real-time facial emotion detection.
* **Data Processing (Python - OpenCV)**
    * Captures video from a webcam.
    * Uses Haar Cascades for efficient face detection.
    * Pre-processes face images to feed into the model.
* **Chatbot Logic (Python)**
    * `ChatBot` class with pre-defined, emotion-specific responses.
    * Logic to cycle through responses to avoid repetition.
* **User Interface**
    * **Web Application:** Built with **FastAPI**, HTML, CSS, and JavaScript. Uses WebSockets for live updates.
    * **Desktop Application:** A standalone GUI built with **Tkinter** for a native experience.

## Key Insights
* The chatbot successfully interprets seven key emotions: happy, sad, angry, fear, disgust, neutral, and surprise.
* The response logic provides dynamic and supportive feedback tailored to the user's mood.
* The dual interface (web and desktop) demonstrates flexibility in deployment and user access.

## Project Structure

```bash
.
├── README.md               # Project overview, setup, and instructions
├── app_copy.py             # Main Python code for the desktop (Tkinter) application
├── index.html              # Core HTML file for the web application's interface
├── login.html              # HTML page for user login
├── signup.html             # HTML page for user sign-up
└── requirements.txt        # Python dependencies
How to Run the Project
Desktop Application
This project is configured to run the desktop application from the root directory.

Clone Repository:

Bash

git clone [https://github.com/yourusername/mood-detection-chatbot.git](https://github.com/yourusername/mood-detection-chatbot.git)
cd mood-detection-chatbot
Install Python Dependencies:

Bash

pip install -r requirements.txt
Run Application:

Bash

python app_copy.py
Web Application
The web application requires a specific folder structure to run correctly. You can create the necessary static/ and templates/ folders locally and place your HTML files accordingly.

Clone and Install Dependencies: Follow steps 1-2 from the "Desktop Application" section.

Organize Files: Create a templates/ folder and move index.html, login.html, and signup.html into it.

Run the FastAPI Server:

Bash

uvicorn app:app --reload
Future Improvements
Add a feature to train the model with new data.

Integrate a conversational AI model (e.g., using Hugging Face Transformers) for more natural and free-form dialogue.

Create a user profile to track and visualize mood history over time.

Deploy the web application to a cloud platform (e.g., Heroku, AWS).

Author
Jayant Sharma

Data Analytics Enthusiast | VIT Vellore

LinkedIn: https://www.linkedin.com/in/jayant-sharma-b1a684267/

Email: jayant.sharma072004@gmail.com
