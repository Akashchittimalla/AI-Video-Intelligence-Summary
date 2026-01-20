import cv2
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO
import ollama

# --- CONFIGURATION ---
VIDEO_SOURCE = "sample.mp4"
MODEL_YOLO = "yolo11n.pt"
MODEL_VISION = "moondream"
LOG_FILE = "ai_action_log.csv"
FRAME_GAP = 15  # Frames to skip between sequence captures

# --- UTILITY: CSV LOGGING ---
def log_action(video_time, label, reasoning):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Video Time (s)', 'Object', 'AI Reasoning'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            f"{video_time:.2f}", 
            label, 
            reasoning
        ])

# --- CORE ENGINE ---
def run_pipeline(video_path):
    print(f"DEBUG: Initializing Pipeline for {video_path}...")
    
    # Check for models folder
    if not os.path.exists("models"):
        os.makedirs("models")

    # Load YOLO
    model = YOLO(MODEL_YOLO) 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    temporal_buffer = [] 
    is_capturing = False
    all_insights = [] # For final summary

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_video_time = frame_id / fps

        # 1. Detection Phase
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Trigger if person found
        if len(results[0].boxes) > 0 and not is_capturing:
            print(f"Action detected at {current_video_time:.2f}s! Capturing sequence...")
            is_capturing = True

        # 2. Sequence Collection (Beginning, Middle, End)
        if is_capturing:
            temporal_buffer.append(frame)
            
            # Skip frames to ensure the 3 frames show movement
            for _ in range(FRAME_GAP): cap.grab() 

            if len(temporal_buffer) == 3:
                img_paths = []
                for idx, f in enumerate(temporal_buffer):
                    path = f"models/seq_frame_{idx}.jpg"
                    cv2.imwrite(path, f)
                    img_paths.append(path)

                # 3. Reasoning Phase (Local SLM)
                try:
                    prompt = "Describe the action taking place across these chronological frames."
                    response = ollama.chat(
                        model=MODEL_VISION,
                        messages=[{'role': 'user', 'content': prompt, 'images': img_paths}]
                    )
                    insight = response['message']['content'].strip()
                    
                    # 4. Logging Phase
                    log_action(current_video_time, "Person", insight)
                    all_insights.append(insight)
                    print(f"\n[LOGGED] {insight}\n")

                except Exception as e:
                    print(f"Ollama Error: {e}. Is Ollama running?")

                # Reset for next detection
                temporal_buffer = []
                is_capturing = False

    cap.release()

    # 5. FINAL MASTER SUMMARY
    if all_insights:
        print("\n" + "="*40)
        print("GENERATING FINAL VIDEO NARRATIVE...")
        summary_prompt = f"Review these events and provide a 2-sentence summary of the overall activity: {'. '.join(all_insights)}"
        try:
            summary_res = ollama.chat(model=MODEL_VISION, messages=[{'role': 'user', 'content': summary_prompt}])
            final_text = summary_res['message']['content'].strip()
            print(f"FINAL SUMMARY: {final_text}")
            log_action(999, "VIDEO_TOTAL", final_text)
        except:
            print("Summary Error: Ensure Ollama is active.")
    
    print("="*40 + "\nProcessing complete.")

if __name__ == "__main__":
    run_pipeline(VIDEO_SOURCE)