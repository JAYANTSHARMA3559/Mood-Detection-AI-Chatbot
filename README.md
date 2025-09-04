# Mood Detection & AI Chatbot

A real-time emotion detection and AI chatbot application that provides personalized responses based on a user's mood. The project features a desktop GUI application using Tkinter.

## 🚀 Key Features

* **Real-time Mood Detection:** Utilizes a pre-trained **Convolutional Neural Network (CNN)** to analyze facial expressions from a live webcam feed.
* **Intelligent Chatbot:** The chatbot provides dynamic and empathetic responses, offering support and encouragement that is tailored to the user's detected emotion.
* **Desktop Application:** A self-contained graphical user interface (GUI) using **Tkinter**.
* **Modular & Maintainable Code:** The project is designed with a clear, modular structure to separate concerns, making it easy to understand and expand.

## 💻 Technologies Used

* **Backend:** Python, Keras/TensorFlow
* **Computer Vision:** OpenCV, Haar Cascade
* **UI/Frontend:** Tkinter
* **Development Tools:** Git

## 📂 Project Structure

```bash
.
├── app_copy.py             # The Tkinter desktop application code
├── index.html              # The login/signup functionality (not used by the main app)
├── login.html              # The login page
├── requirements.txt        # Python dependencies
├── signup.html             # The signup page
└── README.md               # You are here!
⚙️ Setup and Running
Before you start, you must install the necessary Python dependencies. It is highly recommended to use a virtual environment.

Clone the repository:

Bash

git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
Install dependencies:

Bash

pip install -r requirements.txt
Obtain the model:
This repository does not include the large emotiondetector.h5 model file. You will need to train your own model or download it from a separate source.

Run the desktop application:

Bash

python app_copy.py
🤝 Contribution
Contributions are welcome! If you find a bug or have an idea for a new feature, feel free to open an issue or submit a pull request.
