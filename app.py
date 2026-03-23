import streamlit as st
import cv2
import os
import tempfile
import time
from fpdf import FPDF
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ultralytics import YOLO

# --- 1. SETUP & KEYS ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

st.set_page_config(page_title="Vision Intelligence", page_icon="🎬", layout="wide")

if not api_key:
    st.error("⚠️ API Key missing! Set GOOGLE_API_KEY in .env or Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. MODELS ---
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n.pt")

yolo_model = load_yolo()

# --- 3. UTILITY FUNCTIONS ---
def create_pdf_report(report_text):
    """Generates a PDF from the narrative text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Vision Intelligence Narrative Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    # multi_cell wraps text automatically
    pdf.multi_cell(0, 10, txt=report_text)
    return pdf.output(dest='S').encode('latin-1')

def process_yolo_stream(video_path, placeholder, conf):
    """Local YOLOv11 object tracking."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = yolo_model(frame, conf=conf)
        annotated = results[0].plot()
        placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    cap.release()

def generate_ai_narrative(video_path):
    """Gemini 2.5 Flash narrative with 429 Retry Logic."""
    max_retries = 3
    delay = 10
    for i in range(max_retries):
        try:
            with st.status(f"AI Analyzing (Attempt {i+1})...") as status:
                video_file = client.files.upload(file=video_path)
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = client.files.get(name=video_file.name)
                
                # Using the latest stable thinking model
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[video_file, "Analyze this video. List actions with timestamps and a summary."]
                )
                client.files.delete(name=video_file.name)
                return response.text
        except Exception as e:
            if "429" in str(e) and i < max_retries - 1:
                st.warning(f"Rate limited. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else: raise e

# --- 4. MAIN INTERFACE ---
st.sidebar.title("🛠️ Controls")
conf_level = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.4)
tracking_on = st.sidebar.checkbox("Enable YOLO Tracking", value=True)

st.title("🎬 Vision Intelligence Dashboard")
uploaded_file = st.file_uploader("Upload Video Asset", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visual Analysis")
        placeholder = st.empty()
        if tracking_on:
            if st.button("▶️ Start Tracking"):
                process_yolo_stream(video_path, placeholder, conf_level)
        else:
            st.video(uploaded_file)

    with col2:
        st.subheader("Intelligence Report")
        if st.button("🧠 Generate Narrative"):
            report = generate_ai_narrative(video_path)
            st.session_state['report'] = report
            st.markdown(report)
        
        if 'report' in st.session_state:
            pdf_data = create_pdf_report(st.session_state['report'])
            st.download_button("📥 Download PDF Report", pdf_data, "report.pdf", "application/pdf")

    os.remove(video_path)
else:
    st.info("Upload a video to start.")