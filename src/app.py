import streamlit as st
import cv2
import os
import tempfile
import pandas as pd
from ultralytics import YOLO
import ollama
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="AI Video Intel", layout="wide", page_icon="🎥")
st.title("🎥 AI Video Intelligence Pipeline")

# --- SESSION STATE INITIALIZATION ---
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings & Tools")
    if st.button("🗑️ Clear All Data"):
        st.session_state.logs = []
        st.session_state.summary = ""
        st.rerun()

# --- CORE ENGINE ---
def process_video(video_path):
    model_yolo = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    progress_bar = st.progress(0)
    frame_display = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temporal_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_id / fps
        
        # YOLO Detection Trigger
        results = model_yolo.predict(frame, conf=0.5, verbose=False)
        
        if len(results[0].boxes) > 0:
            temporal_buffer.append(frame)
            for _ in range(15): cap.grab() # Skip frames for motion

            if len(temporal_buffer) == 3:
                frame_display.image(temporal_buffer[1], channels="BGR", width=400)
                
                # Save frames for AI
                img_paths = []
                for idx, f in enumerate(temporal_buffer):
                    p = f"models/ui_frame_{idx}.jpg"
                    cv2.imwrite(p, f)
                    img_paths.append(p)

                # AI Reasoning
                try:
                    res = ollama.chat(model='moondream', messages=[{
                        'role': 'user', 'content': 'What is happening in this sequence?', 'images': img_paths
                    }])
                    insight = res['message']['content'].strip()
                    st.session_state.logs.append({"Timestamp": f"{current_time:.1f}s", "Action": insight})
                    st.write(f"🔍 **{current_time:.1f}s:** {insight}")
                except:
                    st.error("Check if Ollama is running!")

                temporal_buffer = []
        
        progress_bar.progress(frame_id / total_frames)
    cap.release()

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    if st.button("🚀 Run Intelligence Pipeline"):
        if not os.path.exists("models"): os.makedirs("models")
        process_video(tfile.name)
        
        # Final Summary Generation
        if st.session_state.logs:
            all_text = " ".join([l['Action'] for l in st.session_state.logs])
            summary_res = ollama.chat(model='moondream', messages=[{
                'role': 'user', 'content': f"Summarize these events into a report: {all_text}"
            }])
            st.session_state.summary = summary_res['message']['content']

# --- RESULTS DISPLAY ---
if st.session_state.summary:
    st.divider()
    st.subheader("📝 Master Narrative")
    st.success(st.session_state.summary)
    
    # Download Button
    df = pd.DataFrame(st.session_state.logs)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Action Log", data=csv, file_name="ai_report.csv", mime="text/csv")