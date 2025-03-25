import cv2
import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
import smtplib
from datetime import datetime
from deepface import DeepFace
from transformers import pipeline

# Load emotion detection models
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
speech_emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", framework="pt")

# Function for real-time facial emotion detection
def detect_emotion_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture image")
        return None, None
    
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        emotion = None
    
    return frame, emotion

# Function for text-based emotion detection
def detect_emotion_text(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']

# Function for speech-based emotion detection
def detect_emotion_speech(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    if not isinstance(audio, np.ndarray):
        raise TypeError("Audio input must be a NumPy array")
    audio = audio.astype(np.float32)
    result = speech_emotion_classifier(audio)
    return result[0]['label']

# Function to recommend tasks based on mood
def recommend_task(emotion):
    task_mapping = {
        "happy": "Work on creative projects",
        "sad": "Take a break or engage in light tasks",
        "anger": "Practice mindfulness or take a walk",
        "fear": "Focus on planning and organizing tasks",
    }
    return task_mapping.get(emotion, "No specific recommendation")

# Function to send email alerts to HR
def send_email(subject, body):
    sender_email = "your_email@example.com"
    receiver_email = "hr@example.com"
    password = "your_password"
    message = f"Subject: {subject}\n\n{body}"
    
    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# Stress management alert system
def stress_alert(name, text):
    result, confidence = detect_emotion_text(text)
    st.write(f"{name} seems {result} (Confidence: {confidence:.2f})")
    
    if result in ['anger', 'sadness', 'fear'] and confidence > 0.5:
        st.warning(f"ALERT: {name} has shown signs of stress! Notify HR.")
        send_email("Stress Alert", f"{name} is experiencing prolonged stress. Please take action.")

# Team Mood Analytics
def team_mood_analysis(team_data):
    mood_counts = {mood: team_data.count(mood) for mood in set(team_data)}
    st.write("\nTeam Mood Analysis:")
    
    fig, ax = plt.subplots()
    ax.bar(mood_counts.keys(), mood_counts.values())
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("Team Mood Analysis")
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Employee Emotion and Mood Analysis System")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option:", [
        "Real-Time Emotion Detection", 
        "Stress Management Alert", 
        "Team Mood Analytics"
    ])

    if choice == "Real-Time Emotion Detection":
        st.header("Real-Time Emotion Detection")
        
        text_input = st.text_input("Enter a message for text emotion detection:")
        if text_input:
            emotion, confidence = detect_emotion_text(text_input)
            st.write(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
            st.write(f"Recommended Task: {recommend_task(emotion)}")
        
        if st.button("Capture and Analyze Facial Emotion"):
            frame, emotion = detect_emotion_camera()
            if frame is not None:
                st.image(frame, channels="BGR", caption="Captured Image")
                if emotion:
                    st.write(f"Detected Emotion: {emotion}")
                    st.write(f"Recommended Task: {recommend_task(emotion)}")
        
        audio_file = st.file_uploader("Upload an audio file for speech emotion detection:", type=["wav"])
        if audio_file:
            emotion = detect_emotion_speech(audio_file)
            st.write(f"Detected Emotion: {emotion}")
            st.write(f"Recommended Task: {recommend_task(emotion)}")

    elif choice == "Stress Management Alert":
        st.header("Stress Management Alert")
        name = st.text_input("Enter employee name:")
        text_input = st.text_input("Enter a message to analyze stress:")
        if name and text_input:
            stress_alert(name, text_input)

    elif choice == "Team Mood Analytics":
        st.header("Team Mood Analytics")
        team_data = st.session_state.get("team_data", [])
        text_input = st.text_input("Enter team message:")
        if text_input:
            emotion, _ = detect_emotion_text(text_input)
            team_data.append(emotion)
            st.session_state.team_data = team_data
        if team_data:
            team_mood_analysis(team_data)

if __name__ == "__main__":
    main()