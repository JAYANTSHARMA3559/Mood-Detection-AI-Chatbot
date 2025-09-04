import asyncio
import os
import random
import time
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from keras.models import load_model


PRIMARY_COLOR = "#4a6fa5"
SECONDARY_COLOR = "#166088"
ACCENT_COLOR = "#4eb17f"
BACKGROUND_COLOR = "#f5f7fa"
TEXT_COLOR = "#333333"
CARD_BG = "#ffffff"

MODEL_PATH = "emotiondetector.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file {MODEL_PATH} not found. Emotion detection will not work.")
    MODEL_EXISTS = False
else:
    MODEL_EXISTS = True

    model = load_model(MODEL_PATH)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


emotion_colors = {
    'angry': '#FF5733',     # Red
    'disgust': '#6E8B3D',   # Olive
    'fear': '#800080',      # Purple
    'happy': '#FFD700',     # Gold
    'neutral': '#A9A9A9',   # Grey
    'sad': '#4682B4',       # Steel Blue
    'surprise': '#FF69B4'   # Hot Pink
}

# Simple in-memory user database
users = {}

# ---------- Feature Extraction ----------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Convert hex color to BGR for OpenCV
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR format

# ---------- ChatBot Class with Pre-defined Responses ----------
class ChatBot:
    def __init__(self):
        print("Initializing ChatBot with pre-defined responses...")
        # Create emotion-specific response templates
        self.responses = {
            "angry": [
                "I notice you seem angry. Taking a few deep breaths might help you feel calmer.",
                "It's okay to feel angry sometimes. Would you like to talk about what's bothering you?",
                "When you're angry, try counting to ten before reacting. It often helps clear your mind.",
                "Your anger is valid, but remember not to let it control you. How about a quick break?",
                "Anger is often a signal that something needs attention. What might it be telling you?",
                "Sometimes writing down what's making you angry can help process the emotion better."
            ],
            "disgust": [
                "I see you're expressing disgust. Let's focus on something more pleasant instead.",
                "Sometimes things can be unpleasant. Would you like to talk about something nicer?",
                "When something disgusts you, it helps to shift your attention elsewhere. What's something you enjoy?",
                "Your reaction is completely natural. Let's move on to something that makes you feel better.",
                "Disgust helps protect us from harmful things. What positive purpose might it serve right now?",
                "Let's take a moment to think about something that brings you joy instead."
            ],
            "fear": [
                "I can see you might be feeling afraid. Remember that you're safe right now.",
                "Fear is your body's way of protecting you, but sometimes it overreacts. Try taking slow, deep breaths.",
                "When you're scared, grounding techniques can help. Try naming five things you can see right now.",
                "It's okay to feel scared. Would talking about it help you process these feelings?",
                "Fear often shrinks when we face it directly. Is there a small step you could take?",
                "Remember that courage isn't the absence of fear, but acting despite it."
            ],
            "happy": [
                "Your smile is contagious! It's wonderful to see you happy today.",
                "Happiness looks good on you! What's bringing you joy right now?",
                "I love seeing you happy! Your positive energy brightens the day.",
                "That smile tells me you're in a good mood. Enjoy these happy moments!",
                "Happiness is a wonderful emotion to share with others. Who might you connect with today?",
                "When we're happy, it's a great time to express gratitude. What are you thankful for?"
            ],
            "neutral": [
                "How are you feeling today? I'm here if you want to talk about anything.",
                "Sometimes a neutral mood is a good reset. Is there anything specific on your mind?",
                "You seem pretty balanced right now. That's a good state for making decisions.",
                "Having a calm day? Sometimes those are the best kind of days.",
                "A neutral mood can be a great time for reflection. Any thoughts you'd like to explore?",
                "Balance is important in life. What helps you maintain yours?"
            ],
            "sad": [
                "I notice you seem a bit down. Remember that it's okay to feel sad sometimes.",
                "Sadness is a natural emotion. Would you like to talk about what's making you feel this way?",
                "When you're feeling sad, sometimes doing something small that you enjoy can help a little.",
                "I'm sorry you're feeling sad. Remember that difficult feelings do pass with time.",
                "Sadness often comes when we value something deeply. What matters to you right now?",
                "Be gentle with yourself when feeling sad. What self-care might help you today?"
            ],
            "surprise": [
                "You look surprised! Did something unexpected happen?",
                "That expression of surprise is quite noticeable! What caught you off guard?",
                "Surprises keep life interesting, don't they? I hope it was a good surprise!",
                "Your surprised reaction makes me curious about what you just discovered!",
                "Surprise can open us to new possibilities. What might this moment be teaching you?",
                "The best surprises often lead to new insights. Any new thoughts coming to mind?"
            ]
        }
        self.used_responses = {emotion: [] for emotion in self.responses.keys()}
        self.last_response_time = 0
        self.current_emotion = None
        self.response_cycle_time = 5  # seconds between response cycles

    def get_response(self, emotion, force_new=False):
        print(f"Generating response for emotion: {emotion}")
        current_time = time.time()
        
        # Change emotion or force new response
        if emotion != self.current_emotion or force_new:
            self.current_emotion = emotion
            self.used_responses[emotion] = []  # Reset used responses for this emotion
        
        try:
            # Get available responses for the emotion
            all_responses = self.responses.get(emotion, self.responses["neutral"])
            available_responses = [r for r in all_responses if r not in self.used_responses[emotion]]
            
            # If all responses have been used, reset
            if not available_responses:
                self.used_responses[emotion] = []
                available_responses = all_responses
            
            # Select a response
            response = random.choice(available_responses)
            self.used_responses[emotion].append(response)
            self.last_response_time = current_time
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I notice you seem {emotion}. How can I help you today?"

    def should_cycle_response(self):
        """Check if it's time to cycle to a new response for the current emotion"""
        current_time = time.time()
        if (current_time - self.last_response_time >= self.response_cycle_time and 
            self.current_emotion is not None):
            return True
        return False

# ---------- Main Application ----------
class EmotionChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mood Detection With Chatbot")
        self.root.geometry("1000x700")
        self.root.configure(bg=BACKGROUND_COLOR)
        
        # Initialize chatbot
        self.chatbot = ChatBot()
        
        # Current emotion and response
        self.current_emotion = "neutral"
        self.current_response = "I'm analyzing your emotions to provide a personalized response..."
        
        # Setup custom fonts and styles
        self.setup_styles()
        
        # Create frames for different screens
        self.main_container = tk.Frame(root, bg=BACKGROUND_COLOR)
        self.main_container.pack(fill="both", expand=True)
        
        # Create frames
        self.login_frame = self.create_login_frame()
        self.signup_frame = self.create_signup_frame()
        self.main_app_frame = self.create_main_app_frame()
        
        # Webcam variables
        self.cap = None
        self.webcam_active = False
        self.stop_threads = False
        
        # Initially show login frame
        self.show_frame(self.login_frame)

    def setup_styles(self):
        # Configure fonts and styles
        self.title_font = ("Roboto", 22, "bold")
        self.subtitle_font = ("Roboto", 12)
        self.button_font = ("Roboto", 11, "bold")
        self.label_font = ("Roboto", 11)
        self.entry_font = ("Roboto", 11)
        
        # Custom style for buttons
        self.button_style = {
            "bg": ACCENT_COLOR,
            "fg": "white",
            "activebackground": "#3c9e69",
            "activeforeground": "white",
            "font": self.button_font,
            "borderwidth": 0,
            "padx": 15,
            "pady": 8,
            "cursor": "hand2"
        }
        
        # Custom style for entries
        self.entry_style = {
            "bg": "white",
            "font": self.entry_font,
            "bd": 1,
            "relief": "solid"
        }

    def create_login_frame(self):
        frame = tk.Frame(self.main_container, bg=BACKGROUND_COLOR)
        
        # Header
        header_frame = tk.Frame(frame, bg=BACKGROUND_COLOR)
        header_frame.pack(pady=30)
        
        title = tk.Label(header_frame, text="Mood Detection with Chatbot", 
                        font=self.title_font, fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR)
        title.pack()
        
        subtitle = tk.Label(header_frame, text="Real-time emotion detection with personalized responses", 
                          font=self.subtitle_font, fg=SECONDARY_COLOR, bg=BACKGROUND_COLOR)
        subtitle.pack(pady=5)
        
        # Form container with white background and shadow effect
        form_container = tk.Frame(frame, bg=CARD_BG, bd=1, relief="solid")
        form_container.pack(padx=30, pady=20)
        
        # Form title
        form_title = tk.Label(form_container, text="Login", 
                            font=("Roboto", 16, "bold"), fg=SECONDARY_COLOR, bg=CARD_BG)
        form_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=30, pady=(30, 20))
        
        # Error message
        self.login_error = tk.Label(form_container, text="", fg="red", bg=CARD_BG, font=self.label_font)
        self.login_error.grid(row=1, column=0, columnspan=2, sticky="ew", padx=30, pady=(0, 10))
        
        # Username
        username_label = tk.Label(form_container, text="Username", 
                                font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        username_label.grid(row=2, column=0, sticky="w", padx=30, pady=(10, 5))
        
        self.login_username = tk.Entry(form_container, width=40, **self.entry_style)
        self.login_username.grid(row=3, column=0, padx=30, pady=(0, 15))
        
        # Password
        password_label = tk.Label(form_container, text="Password", 
                                font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        password_label.grid(row=4, column=0, sticky="w", padx=30, pady=(0, 5))
        
        self.login_password = tk.Entry(form_container, width=40, show="•", **self.entry_style)
        self.login_password.grid(row=5, column=0, padx=30, pady=(0, 20))
        
        # Login button
        login_button = tk.Button(form_container, text="Login", 
                              command=self.login_user, **self.button_style)
        login_button.grid(row=6, column=0, sticky="ew", padx=30, pady=(10, 20))
        
        # Link to signup
        signup_link = tk.Label(form_container, text="Don't have an account? Sign up", 
                             fg=PRIMARY_COLOR, bg=CARD_BG, cursor="hand2")
        signup_link.grid(row=7, column=0, sticky="w", padx=30, pady=(0, 30))
        signup_link.bind("<Button-1>", lambda e: self.show_frame(self.signup_frame))
        
        return frame
    
    def create_signup_frame(self):
        frame = tk.Frame(self.main_container, bg=BACKGROUND_COLOR)
        
        # Header
        header_frame = tk.Frame(frame, bg=BACKGROUND_COLOR)
        header_frame.pack(pady=30)
        
        title = tk.Label(header_frame, text="Mood Detection with Chatbot", 
                       font=self.title_font, fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR)
        title.pack()
        
        subtitle = tk.Label(header_frame, text="Real-time emotion detection with personalized responses", 
                          font=self.subtitle_font, fg=SECONDARY_COLOR, bg=BACKGROUND_COLOR)
        subtitle.pack(pady=5)
        
        # Form container
        form_container = tk.Frame(frame, bg=CARD_BG, bd=1, relief="solid")
        form_container.pack(padx=30, pady=20)
        
        # Form title
        form_title = tk.Label(form_container, text="Sign Up", 
                            font=("Roboto", 16, "bold"), fg=SECONDARY_COLOR, bg=CARD_BG)
        form_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=30, pady=(30, 20))
        
        # Error message
        self.signup_error = tk.Label(form_container, text="", fg="red", bg=CARD_BG, font=self.label_font)
        self.signup_error.grid(row=1, column=0, columnspan=2, sticky="ew", padx=30, pady=(0, 10))
        
        # Username
        username_label = tk.Label(form_container, text="Username", 
                                font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        username_label.grid(row=2, column=0, sticky="w", padx=30, pady=(10, 5))
        
        self.signup_username = tk.Entry(form_container, width=40, **self.entry_style)
        self.signup_username.grid(row=3, column=0, padx=30, pady=(0, 15))
        
        # Email
        email_label = tk.Label(form_container, text="Email", 
                             font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        email_label.grid(row=4, column=0, sticky="w", padx=30, pady=(0, 5))
        
        self.signup_email = tk.Entry(form_container, width=40, **self.entry_style)
        self.signup_email.grid(row=5, column=0, padx=30, pady=(0, 15))
        
        # Password
        password_label = tk.Label(form_container, text="Password", 
                                font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        password_label.grid(row=6, column=0, sticky="w", padx=30, pady=(0, 5))
        
        self.signup_password = tk.Entry(form_container, width=40, show="•", **self.entry_style)
        self.signup_password.grid(row=7, column=0, padx=30, pady=(0, 15))
        
        # Confirm Password
        confirm_label = tk.Label(form_container, text="Confirm Password", 
                               font=self.label_font, fg=SECONDARY_COLOR, bg=CARD_BG)
        confirm_label.grid(row=8, column=0, sticky="w", padx=30, pady=(0, 5))
        
        self.signup_confirm = tk.Entry(form_container, width=40, show="•", **self.entry_style)
        self.signup_confirm.grid(row=9, column=0, padx=30, pady=(0, 20))
        
        # Sign Up button
        signup_button = tk.Button(form_container, text="Sign Up", 
                               command=self.register_user, **self.button_style)
        signup_button.grid(row=10, column=0, sticky="ew", padx=30, pady=(10, 20))
        
        # Link to login
        login_link = tk.Label(form_container, text="Already have an account? Login", 
                            fg=PRIMARY_COLOR, bg=CARD_BG, cursor="hand2")
        login_link.grid(row=11, column=0, sticky="w", padx=30, pady=(0, 30))
        login_link.bind("<Button-1>", lambda e: self.show_frame(self.login_frame))
        
        return frame
    
    def create_main_app_frame(self):
        frame = tk.Frame(self.main_container, bg=BACKGROUND_COLOR)
        
        # Header with title and logout button
        header = tk.Frame(frame, bg=BACKGROUND_COLOR)
        header.pack(fill="x", padx=20, pady=10)
        
        title = tk.Label(header, text="🧠 Mood Detection & ChatBot", 
                       font=self.title_font, fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR)
        title.pack(side="left")
        
        logout_btn = tk.Button(header, text="Logout", command=self.logout, **self.button_style)
        logout_btn.pack(side="right")
        
        # Content area
        content = tk.Frame(frame, bg=BACKGROUND_COLOR)
        content.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Video section (left)
        video_section = tk.Frame(content, bg=BACKGROUND_COLOR)
        video_section.pack(side="left", fill="both", expand=True)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_section, width=640, height=480, bg="black", highlightthickness=0)
        self.video_canvas.pack(pady=10)
        
        # Emotion display section (right)
        emotion_section = tk.Frame(content, bg=BACKGROUND_COLOR)
        emotion_section.pack(side="right", fill="both", expand=True, padx=20)
        
        # Emotion box
        emotion_box = tk.Frame(emotion_section, bg=CARD_BG, bd=1, relief="solid")
        emotion_box.pack(fill="x", pady=10, padx=10)
        
        # Emotion display
        emotion_label = tk.Label(emotion_box, text="Detected Emotion:", 
                               font=("Roboto", 14, "bold"), fg=SECONDARY_COLOR, bg=CARD_BG)
        emotion_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        emotion_display = tk.Frame(emotion_box, bg=CARD_BG)
        emotion_display.pack(fill="x", padx=15, pady=(0, 15))
        
        self.emotion_indicator = tk.Canvas(emotion_display, width=20, height=20, 
                                         bg="#ccc", highlightthickness=0)
        self.emotion_indicator.pack(side="left", padx=(0, 10))
        
        self.emotion_text = tk.Label(emotion_display, text="Waiting for emotion detection...", 
                                   font=self.label_font, bg=CARD_BG)
        self.emotion_text.pack(side="left")
        
        # Response section
        response_label = tk.Label(emotion_box, text="ChatBot Response:", 
                                font=("Roboto", 14, "bold"), fg=SECONDARY_COLOR, bg=CARD_BG)
        response_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        response_bubble = tk.Frame(emotion_box, bg="#E6F7FF", bd=0)
        response_bubble.pack(fill="x", padx=15, pady=(0, 15))
        
        self.response_text = tk.Text(response_bubble, height=8, wrap="word", 
                                   font=self.label_font, bg="#E6F7FF", bd=0,
                                   padx=10, pady=10)
        self.response_text.pack(fill="both", expand=True)
        self.response_text.insert("1.0", self.current_response)
        self.response_text.config(state="disabled")
        
        # Emotion history
        history_box = tk.Frame(emotion_section, bg=CARD_BG, bd=1, relief="solid")
        history_box.pack(fill="x", pady=10, padx=10)
        
        history_label = tk.Label(history_box, text="Emotion History", 
                               font=("Roboto", 14, "bold"), fg=SECONDARY_COLOR, bg=CARD_BG)
        history_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Scrollable history list
        history_frame = tk.Frame(history_box, bg=CARD_BG)
        history_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.history_listbox = tk.Listbox(history_frame, height=6, 
                                        font=self.label_font, bg="#f9f9f9", bd=1)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(history_frame, orient="vertical", 
                               command=self.history_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        
        self.history_listbox.config(yscrollcommand=scrollbar.set)
        self.history_listbox.insert(tk.END, "Waiting for emotions...")
        
        return frame
    
    def show_frame(self, frame):
        # Hide all frames
        for f in [self.login_frame, self.signup_frame, self.main_app_frame]:
            f.pack_forget()
        
        # Show the requested frame
        frame.pack(fill="both", expand=True)
        
        # Start webcam if showing main app
        if frame == self.main_app_frame and not self.webcam_active:
            self.start_webcam()
        # Stop webcam if leaving main app
        elif frame != self.main_app_frame and self.webcam_active:
            self.stop_webcam()
    
    def login_user(self):
        username = self.login_username.get()
        password = self.login_password.get()
        
        if not username or not password:
            self.login_error.config(text="Please enter both username and password")
            return
        
        if username in users and users[username]["password"] == password:
            # Successful login
            self.login_error.config(text="")
            self.show_frame(self.main_app_frame)
        else:
            self.login_error.config(text="Invalid username or password")
    
    def register_user(self):
        username = self.signup_username.get()
        email = self.signup_email.get()
        password = self.signup_password.get()
        confirm = self.signup_confirm.get()
        
        # Basic validation
        if not username or not email or not password or not confirm:
            self.signup_error.config(text="All fields are required")
            return
        
        if password != confirm:
            self.signup_error.config(text="Passwords do not match")
            return
        
        if username in users:
            self.signup_error.config(text="Username already exists")
            return
        
        # Register user
        users[username] = {
            "email": email,
            "password": password
        }
        
        # Show success message
        messagebox.showinfo("Success", "Registration successful! Please login with your new account.")
        
        # Clear fields
        self.signup_username.delete(0, tk.END)
        self.signup_email.delete(0, tk.END)
        self.signup_password.delete(0, tk.END)
        self.signup_confirm.delete(0, tk.END)
        
        # Switch to login
        self.show_frame(self.login_frame)
    
    def logout(self):
        # Stop webcam
        self.stop_webcam()
        
        # Clear login fields
        self.login_username.delete(0, tk.END)
        self.login_password.delete(0, tk.END)
        
        # Show login frame
        self.show_frame(self.login_frame)
    
    def start_webcam(self):
        if not MODEL_EXISTS:
            messagebox.showwarning("Warning", 
                                 "Emotion detection model not found. Using simulated emotions.")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.webcam_active = True
        self.stop_threads = False
        
        # Start emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotion)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()
    
    def stop_webcam(self):
        self.webcam_active = False
        self.stop_threads = True
        
        # Release resources
        if self.cap:
            self.cap.release()
    
    def detect_emotion(self):
        last_emotion = None
        last_response_time = 0
        
        while self.webcam_active and not self.stop_threads:
            try:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # If model doesn't exist, simulate emotions
                if not MODEL_EXISTS:
                    time.sleep(3)
                    simulated_emotions = list(labels.values())
                    current_emotion = random.choice(simulated_emotions)
                    
                    # Draw a fake detection rectangle in the center of frame
                    h, w = display_frame.shape[:2]
                    fake_x, fake_y = w//4, h//4
                    fake_w, fake_h = w//2, h//2
                    cv2.rectangle(display_frame, (fake_x, fake_y), (fake_x + fake_w, fake_y + fake_h), (0, 255, 0), 2)
                    
                    # Convert hex color to BGR
                    color = emotion_colors.get(current_emotion, "#FFFFFF")
                    bgr_color = hex_to_bgr(color)
                    
                    # Add emotion text
                    cv2.putText(display_frame, current_emotion, (fake_x, fake_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr_color, 2)
                    
                    # Update UI
                    self.update_emotion(current_emotion)
                    
                    # Show the frame
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.update_frame(rgb_frame)
                    
                    time.sleep(3)  # Wait longer for simulated emotions
                    continue
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                current_emotion = None
                for i, (x, y, w, h) in enumerate(faces):
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Extract face region for emotion detection
                    face_img = gray[y:y+h, x:x+w]
                    resized_face = cv2.resize(face_img, (48, 48))
                    processed_face = extract_features(resized_face)
                    
                    # Get emotion prediction
                    prediction = model.predict(processed_face, verbose=0)
                    emotion = labels[np.argmax(prediction)]
                    
                    # Only update UI with first face emotion
                    if i == 0:
                        current_emotion = emotion
                    
                    # Convert hex color to BGR
                    color = emotion_colors.get(emotion, "#FFFFFF")
                    bgr_color = hex_to_bgr(color)
                    
                    # Add emotion text
                    cv2.putText(display_frame, emotion, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr_color, 2)
                
                # Update UI with detected emotion if available
                if current_emotion is not None:
                    current_time = time.time()
                    if current_emotion != last_emotion:
                        self.update_emotion(current_emotion)
                        last_emotion = current_emotion
                        last_response_time = current_time
                    # Cycle response if it's time
                    elif current_time - last_response_time > 5:
                        self.update_response(
                            self.chatbot.get_response(current_emotion, force_new=True)
                        )
                        last_response_time = current_time
                
                # Convert to RGB for tkinter display
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Update UI on main thread
                self.update_frame(rgb_frame)
            
            except Exception as e:
                print(f"Error in emotion detection: {e}")
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_frame(self, frame):
        """Update the video canvas with the provided frame"""
        # Convert to PhotoImage
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.image = imgtk  # Keep a reference
    
    def update_emotion(self, emotion):
        # Update emotion display
        color = emotion_colors.get(emotion, "#ccc")
        
        # Update on UI thread
        self.root.after(0, lambda: self.emotion_indicator.config(bg=color))
        self.root.after(0, lambda: self.emotion_text.config(text=f"Current emotion: {emotion}"))
        
        # Update history
        timestamp = time.strftime("%H:%M:%S")
        
        def update_history():
            # Clear placeholder if needed
            if self.history_listbox.size() == 1 and self.history_listbox.get(0) == "Waiting for emotions...":
                self.history_listbox.delete(0)
            
            # Add new detection at the top
            self.history_listbox.insert(0, f"{emotion} - {timestamp}")
            
            # Keep only recent history
            if self.history_listbox.size() > 10:
                self.history_listbox.delete(10)
        
        self.root.after(0, update_history)
        
        # Get and update response
        response = self.chatbot.get_response(emotion)
        self.update_response(response)
    
    def update_response(self, response):
        def update_text():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(1.0, response)
            self.response_text.config(state=tk.DISABLED)
        
        self.root.after(0, update_text)

# Entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionChatbotApp(root)
    root.mainloop()
