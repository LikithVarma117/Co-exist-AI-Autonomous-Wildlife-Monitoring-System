import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tempfile
import os
import pandas as pd
from pathlib import Path
import io
import shutil
import time
from collections import defaultdict
import datetime
import requests
from urllib.parse import urlencode
import subprocess # For ffmpeg
import threading
import serial # For ESP32 Communication
import serial.tools.list_ports # To find the ESP32 automatically
import yaml

# Required for HTML embedding of videos in Streamlit
from base64 import b64encode
import base64
from streamlit.components.v1 import html

# SpeciesNet imports
from speciesnet import SpeciesNet
import kagglehub

# ResNet50 imports
import timm 
import torch

from twilio.rest import Client
import smtplib
from email.message import EmailMessage

# --- Google Gemini Import ---
import google.generativeai as genai

# --- DeepSORT Imports ---
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# --- REAL-TIME WEBCAM IMPORTS ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Co-Exist AI : Autonomous Wildlife Monitoring System", page_icon="🐾", layout="wide", initial_sidebar_state="expanded")

st.title("Co-Exist AI : Autonomous Wildlife Monitoring System")
st.markdown("""
<style>
    .st-emotion-cache-1r6dm7m { /* Targets the main markdown area */
        font-size: 1.1em;
    }
</style>
Image Mode: Uses MegaDetector + SpeciesNet or YOLOv11 + ResNet50. **Educational Tool & Chatbot.**
Video Mode: Uses YOLOv11. **Alerts ONLY for wild animals.**
Webcam Mode: **Autonomous Real-Time.** **Alerts ONLY for wild animals with Population Count.**
Alerts: Enable SMS, Voice Call, Email, and Telegram notifications in the sidebar.
""", unsafe_allow_html=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- TRACKER FIX: CREATE CUSTOM CONFIG FOR STICKY TRACKING ---
def create_tracker_config():
    """Creates a custom tracker config to prevent ID switching/flickering."""
    config_content = """
    tracker_type: bytetrack
    track_high_thresh: 0.5
    track_low_thresh: 0.1
    new_track_thresh: 0.6
    track_buffer: 120  # Increased to 120 frames (4 seconds) memory
    match_thresh: 0.8
    """
    if not os.path.exists("custom_bytetrack.yaml"):
        with open("custom_bytetrack.yaml", "w") as f:
            f.write(config_content)

create_tracker_config()

# --- CONFIGURATION FOR GEMINI ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("⚠️ GOOGLE_API_KEY missing in .streamlit/secrets.toml")
except Exception as e:
    st.warning(f"Gemini Config Error: {e}")

# --- STRICT ANIMAL FILTERING CONFIGURATION (For Video/Webcam) ---
TARGET_ANIMAL_CLASSES = ['elephant', 'bear', 'zebra', 'giraffe', 'cow', 'buffalo', 'rhino', 'lion', 'leopard', 'tiger', 'deer'] 
EXCLUDED_CLASSES_FOR_ALERTS = ['bird','cat', 'dog', 'sheep','horse'] 

# --- IOT CONFIGURATION (ESP32) ---
ESP32_PORT = "COM3"  
BAUD_RATE = 9600

# --- ROBUST IOT FUNCTION (OPEN-SEND-CLOSE) ---
def send_iot_alert_direct():
    """
    Opens the port, sends 'A', and closes it immediately.
    This prevents the 'Access Denied' error on restart.
    """
    try:
        # 1. Try configured port
        port = ESP32_PORT
        
        # 2. Auto-detect if needed (Optional fallback)
        if not port:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if "CP210" in p.description or "CH340" in p.description or "USB Serial" in p.description:
                    port = p.device
                    break
        
        if port:
            # Context manager handles opening AND closing automatically
            with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
                time.sleep(2.0) # Stabilization wait for ESP32 reboot
                ser.write(b'A') # Send signal
                print(f"✅ [IOT] Triggered LED/Buzzer on {port}")
        else:
            print("⚠️ [IOT] No COM port found.")
            
    except serial.SerialException as e:
        print(f"❌ [IOT] Port Busy/Error: {e}")
    except Exception as e:
        print(f"❌ [IOT] Error: {e}")


# --- ALERT MANAGER ---
class AlertManager:
    def __init__(self):
        self.last_alert_time = {}
        self.cooldown_seconds = 60 

    def should_alert(self, class_name):
        current_time = time.time()
        if class_name not in self.last_alert_time:
            self.last_alert_time[class_name] = current_time
            return True

        if current_time - self.last_alert_time[class_name] > self.cooldown_seconds:
            self.last_alert_time[class_name] = current_time
            return True
        return False

if 'alert_manager' not in st.session_state:
    st.session_state['alert_manager'] = AlertManager()


# --------------------------------------------------------
# MODEL LOADING FUNCTIONS
# --------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_name):
    try:
        is_segmentation_model = False
        if 'seg' not in model_name:
            if "runs/" not in model_name: 
                seg_model_name = model_name.replace('.pt', '-seg.pt')
                try:
                    model = YOLO(seg_model_name)
                    is_segmentation_model = True
                    st.success(f"YOLO '{seg_model_name}' (segmentation) loaded.")
                    return model, is_segmentation_model
                except Exception:
                    pass 

        model = YOLO(model_name)
        if hasattr(model, 'overrides') and model.overrides.get('task') == 'segment':
             is_segmentation_model = True
        elif 'seg' in str(model_name):
            is_segmentation_model = True
            
        st.success(f"YOLO '{os.path.basename(model_name)}' loaded.")
        return model, is_segmentation_model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None, False


@st.cache_resource
def load_speciesnet_model():
    speciesnet_model_path = Path("speciesnet_model")
    speciesnet_model_path.mkdir(exist_ok=True)
    try:
        if not (speciesnet_model_path / 'model_config.json').exists():
            st.info("Downloading SpeciesNet model...");
            download_path = kagglehub.model_download('google/speciesnet/PyTorch/v4.0.1a')
            for item in os.listdir(download_path):
                source, dest = os.path.join(download_path, item), os.path.join(speciesnet_model_path, item)
                if os.path.isdir(source):
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, dest)
            st.success("SpeciesNet model downloaded and copied.")
        model = SpeciesNet(str(speciesnet_model_path))
        st.success("SpeciesNet model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading SpeciesNet: {e}")
        return None


@st.cache_resource
def load_resnet_model(model_name='resnet50', pretrained=True):
    try:
        classifier_model = timm.create_model(model_name, pretrained=pretrained)
        classifier_model = classifier_model.eval().to(DEVICE)
        data_config = timm.data.resolve_model_data_config(classifier_model)
        classifier_transforms = timm.data.create_transform(**data_config, is_training=False)
        return classifier_model, classifier_transforms
    except Exception as e:
        st.error(f"Error loading ResNet classifier model ({model_name}): {e}")
        return None, None

@st.cache_data
def get_imagenet_classes():
    try:
        response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
        response.raise_for_status()
        return response.text.splitlines()
    except Exception as e:
        st.error(f"Could not download ImageNet class names: {e}. Using generic list.")
        return [f"class_{i}" for i in range(1000)]

# --------------------------------------------------------
# GOOGLE GEMINI SETUP
# --------------------------------------------------------
@st.cache_resource
def configure_gemini():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    # Using standard flash model
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_model = configure_gemini()

def generate_animal_info(animal_name):
    if not gemini_model:
        return "⚠️ Google API Key not found in secrets.toml."
    
    prompt = f"""
    Provide a concise and informative summary about the animal: {animal_name}.
    Structure your response in Markdown with the following sections:
    ### Conservation Status
    - Is the animal endangered, extinct, vulnerable, or of least concern?
    ### Interesting Facts
    - List 3-4 bullet points with fascinating facts about this species.
    ### Habitat and Geographic Distribution
    - Describe its primary habitat and list key countries or continents where it is found.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error retrieving info from Google AI: {e}"

# --- CHATBOT FUNCTION ---
def ask_gemini_chatbot(animal_name, user_question):
    if not gemini_model:
        return "⚠️ Google API Key missing."
    
    prompt = f"""
    You are a wildlife expert. The user is looking at an image of a {animal_name}.
    User Question: "{user_question}"
    Answer the question accurately and concisely specifically about the {animal_name}.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
# -------------------------------------------------

# --------------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------------
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🚀 Running on: {gpu_name}")
else:
    st.sidebar.error("⚠️ Running on CPU (Slow)")
    
# Check port
try:
    if serial.tools.list_ports.comports():
        st.sidebar.success(f"🔌 ESP32 Port Detected") 
    else:
        st.sidebar.warning("🔌 ESP32 Not Detected")
except:
    pass

st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Input Mode", ["Image", "Video", "Webcam"])
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.01)

if mode == "Image" and st.session_state.get('image_classifier_choice', 'ResNet50') == "ResNet50":
    iou_threshold = st.sidebar.slider("NMS IOU Threshold (Image Mode)", 0.0, 1.0, 0.4, 0.05, help="Lower value means more aggressive Non-Maximum Suppression.")
else:
    iou_threshold = 0.7


if mode == "Image":
    st.sidebar.subheader("Image Classification Model")
    image_classifier_choice = st.sidebar.selectbox(
        "Choose Classifier for Images", ["SpeciesNet", "ResNet50"], key='image_classifier_choice'
    )
    if image_classifier_choice == "ResNet50":
        st.sidebar.subheader("YOLOv11 Detector for ResNet50 (Image Mode)")
        model_selection_image_resnet = st.sidebar.selectbox(
            "Choose YOLOv11 Model for Detection", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=1
        )


elif mode in ["Video", "Webcam"]:
    st.sidebar.subheader("Detection & Segmentation Model (YOLOv11)")
    
    # --- CHECK FOR CUSTOM MODEL ---
    expected_path = Path("runs/detect/wildlife_coco9_custom/weights/best.pt")
    
    # Standard models
    model_options = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"]
    
    found_models = []
    if expected_path.exists():
        found_models.append(str(expected_path))
    else:
        # Fallback recursive search
        found_models = [str(p) for p in Path(".").rglob("best.pt")]
        found_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if found_models:
        model_options = found_models + model_options
        st.sidebar.success(f"✅ Found Custom Model!")
    else:
        model_options = model_options

    model_selection_video_webcam = st.sidebar.selectbox(
        "Choose YOLOv11 Model", model_options, index=0
    )
    
    st.sidebar.subheader("Object Tracker")
    tracker_type = st.sidebar.selectbox("Select Tracker Algorithm", ["DeepSORT", "ByteTrack"], index=0, help="DeepSORT is slower but handles occlusions well. ByteTrack is faster and robust.")


st.sidebar.header("Alerting")
sms_enabled = st.sidebar.checkbox("Enable SMS Alerts")
call_enabled = st.sidebar.checkbox("Enable Call Alerts")
email_enabled = st.sidebar.checkbox("Enable Email Alerts")
telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts")
iot_enabled = st.sidebar.checkbox("Enable IoT (ESP32) Alerts", value=True)

# --------------------------------------------------------
# ALERT & HELPER FUNCTIONS
# --------------------------------------------------------
def send_iot_alert(serial_connection):
    # This function opens port, sends, and closes immediately to avoid conflicts
    try:
        # Use configured port
        port = ESP32_PORT
        if port:
            with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
                time.sleep(0.1) 
                ser.write(b'A') 
                print(f"✅ [IOT] Signal 'A' sent to {port}")
        else:
            print("⚠️ [IOT] Skipped (No Port)")
    except Exception as e:
        print(f"❌ [IOT] Error: {e}")

def _should_send_alert(summary_df, current_mode):
    if summary_df.empty: return False

    if current_mode == "Video":
        if "Class" in summary_df.columns:
            for cls in summary_df['Class'].unique():
                if cls.lower() in TARGET_ANIMAL_CLASSES:
                    return True
        return False
    
    # Image mode no longer triggers 'alerts', only displays info, so we can return False or handle differently
    # But if you wanted alerts for everything in image mode, you would return True here.
    # Since you asked to REMOVE alerts for image mode, we return False.
    elif current_mode == "Image":
        return False 

    return False

def get_alert_summary(summary_df):
    if summary_df.empty: return ""
    counts = None

    if "Class" in summary_df.columns:
        if mode == "Video":
             relevant_df = summary_df[summary_df['Class'].str.lower().isin(TARGET_ANIMAL_CLASSES)]
        else:
             relevant_df = summary_df 

        if relevant_df.empty: return ""
        if "Total Count" in relevant_df.columns:
            counts_dict = relevant_df.set_index('Class')['Total Count'].to_dict()
            return ", ".join([f"{count} {species}" for species, count in counts_dict.items()])
        else:
            counts = relevant_df['Class'].value_counts()

    elif "SpeciesNet Prediction" in summary_df.columns:
        def is_valid_animal(row):
            return row['Category'] == 'animal'
        relevant_df = summary_df[summary_df.apply(is_valid_animal, axis=1)]
        if relevant_df.empty: return ""
        counts = relevant_df['SpeciesNet Prediction'].value_counts()

    elif "Species" in summary_df.columns:
         counts = summary_df['Species'].value_counts()

    if counts is None or counts.empty: return ""
    return ", ".join([f"{count} {species}" for species, count in counts.items()])

def send_sms_alert(summary_string):
    if not summary_string: return
    print(f"📡 [SMS] Preparing to send alert for: {summary_string}")
    try:
        if "TWILIO_ACCOUNT_SID" not in st.secrets:
            print("❌ [SMS] Error: TWILIO_ACCOUNT_SID not found in secrets.toml")
            return
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        message_body = f"Wildlife Alert: {summary_string} detected."
        client.messages.create(body=message_body, from_=st.secrets["TWILIO_PHONE_NUMBER"], to=st.secrets["RECIPIENT_SMS_NUMBER"])
        print("✅ [SMS] Alert Sent Successfully")
    except Exception as e:
        print(f"❌ [SMS] Failed: {e}")

def send_voice_call_alert(summary_string):
    if not summary_string: return
    print(f"📡 [CALL] Preparing to initiate call for: {summary_string}")
    try:
        if "TWIML_BIN_URL" not in st.secrets:
            print("❌ [CALL] Error: TWIML_BIN_URL not found in secrets.toml")
            return
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        twiml_url = st.secrets["TWIML_BIN_URL"] + "?" + urlencode({"Message": summary_string})
        client.calls.create(url=twiml_url, to=st.secrets["RECIPIENT_CALL_NUMBER"], from_=st.secrets["TWILIO_PHONE_NUMBER"])
        print("✅ [CALL] Initiated Successfully")
    except Exception as e:
        print(f"❌ [CALL] Failed: {e}")

def send_email_alert(summary_string):
    if not summary_string: return
    print(f"📡 [EMAIL] Preparing to send email for: {summary_string}")
    try:
        msg = EmailMessage()
        formatted_body = summary_string.replace(', ', '\n- ')
        msg.set_content(f"Animals detected:\n- {formatted_body}")
        msg['Subject'] = "Wildlife Detection Alert!"
        
        if "RECIPIENT_EMAIL_ADDRESS" in st.secrets:
            msg['To'] = st.secrets["RECIPIENT_EMAIL_ADDRESS"]
            user = st.secrets.get("SENDER_EMAIL", "coexist.ai.wildlife@gmail.com") 
            password = st.secrets.get("SENDER_PASSWORD", "txoq mbeu jtau xgcp") 
        else:
            print("⚠️ [EMAIL] Warning: RECIPIENT_EMAIL_ADDRESS not in secrets, falling back to dummy.")
            msg['To'] = "recipient@example.com"
            user = "coexist.ai.wildlife@gmail.com"
            password = "txoq mbeu jtau xgcp"

        msg['From'] = user
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        print(f"✅ [EMAIL] Sent Successfully to {msg['To']}")
    except Exception as e:
        print(f"❌ [EMAIL] Failed: {e}")

def send_telegram_alert(summary_string, species_frames):
    if not summary_string: return
    print(f"📡 [TELEGRAM] Preparing to send alert for: {summary_string}")
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
             print("❌ [TELEGRAM] Error: Token or Chat ID missing in secrets.toml")
             return

        summary_message = f"Wildlife Alert: {summary_string} detected."
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={'chat_id': chat_id, 'text': summary_message})

        if species_frames:
            photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
            for species, frame_image in species_frames.items():
                image_stream = io.BytesIO(); frame_image.save(image_stream, format='JPEG'); image_stream.seek(0)
                files = {'photo': (f'detected_{species}.jpg', image_stream)}
                requests.post(photo_url, data={'chat_id': chat_id, 'caption': f"Frame showing detected: {species}"}, files=files)
                time.sleep(1)

        print("✅ [TELEGRAM] Sent Successfully")
    except Exception as e:
        print(f"❌ [TELEGRAM] Failed: {e}")

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def save_uploaded_file_from_pil(pil_image):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            pil_image.save(tmp.name, format="JPEG")
            return tmp.name
    except Exception as e:
        st.error(f"Error saving PIL image: {e}")
        return None

def get_species_name(speciesnet_prediction_string):
    parts = speciesnet_prediction_string.split(';')
    meaningful_parts = [p.strip() for p in parts if p.strip() and p.strip() != 'no cv result']
    if not meaningful_parts: return "Unknown Species", "N/A"
    common_name = meaningful_parts[-1].replace('_', ' ').title()
    scientific_name = "N/A"
    if len(meaningful_parts) >= 2 and '_' in meaningful_parts[-2]:
        scientific_name = meaningful_parts[-2].replace('_', ' ').capitalize()
    elif len(meaningful_parts) >= 3:
        genus = meaningful_parts[-3].capitalize()
        specific_epithet = meaningful_parts[-2].lower()
        if genus.lower() != specific_epithet: scientific_name = f"{genus} {specific_epithet}"
        else: scientific_name = genus
    elif len(meaningful_parts) >= 2:
        scientific_name = meaningful_parts[-2].capitalize()
    return common_name, scientific_name

def get_gemini_facts(species_name):
    if not species_name or species_name == "Unknown": return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Give me 3 short, interesting facts about the {species_name} animal. Keep it under 50 words total."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not fetch facts. Error: {str(e)}"

def draw_annotations_on_pil(img_pil, detections, default_font_path="arial.ttf"):
    draw = ImageDraw.Draw(img_pil)
    img_width, img_height = img_pil.size
    padding = max(3, int(img_width / 400))
    line_spacing = max(2, int(img_width / 1000))
    box_thickness = max(2, int(img_width / 300))
    for det in detections:
        x_rel, y_rel, w_rel, h_rel = det["bbox"]
        x1, y1 = int(x_rel * img_width), int(y_rel * img_height)
        x2, y2 = int((x_rel + w_rel) * img_width), int((y_rel + h_rel) * img_height)
        label = det["label"]
        try: font = ImageFont.truetype(default_font_path, max(12, min(min(max(10, int((y2-y1) / 12)), max(10, int((x2-x1) / 15))), 40)))
        except IOError: font = ImageFont.load_default()
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=box_thickness)
        draw.rectangle([(x1, y1-20), (x1+100, y1)], fill=(0, 255, 0))
        draw.text((x1 + padding, y1 + padding), label, font=font, fill=(255, 255, 255))
    return img_pil

def get_species_from_resnet(cropped_image_pil, resnet_model, resnet_transforms, imagenet_classes):
    if cropped_image_pil.size[0] == 0 or cropped_image_pil.size[1] == 0: return "Invalid Crop", 0.0
    input_tensor = resnet_transforms(cropped_image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = resnet_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        classified_species = imagenet_classes[top_catid[0].item()]
        confidence = top_prob[0].item()
    return classified_species, confidence

def process_frame_with_speciesnet(img_pil, speciesnet_model, conf):
    temp_image_path = save_uploaded_file_from_pil(img_pil)
    if not temp_image_path: return img_pil, pd.DataFrame()
    detection_data, annotated_detections = [], []
    try:
        predictions = speciesnet_model.predict(instances_dict={"instances": [{"filepath": temp_image_path, "detector_threshold": conf}]})
        os.remove(temp_image_path)
        if predictions and predictions['predictions']:
            pred_item = predictions['predictions'][0]
            if 'detections' in pred_item and pred_item['detections']:
                for det in pred_item['detections']:
                    if det.get('conf', 0.0) < conf: continue
                    md_cat_raw = det.get('category', '0')
                    md_category = "animal" if md_cat_raw == '1' else "person" if md_cat_raw == '2' else "vehicle" if md_cat_raw == '3' else "other"
                    common_name = md_category.title(); scientific_name = "N/A"
                    if md_category == "animal" and pred_item.get('classifications') and pred_item['classifications']['classes']:
                        top_classification_string = pred_item['classifications']['classes'][0]
                        common_name, scientific_name = get_species_name(top_classification_string)
                    display_label = f"{scientific_name}\n{common_name} ({det['conf']:.2f})" if scientific_name != "N/A" else f"{common_name} ({det['conf']:.2f})"
                    detection_data.append({"Detector": "MegaDetector", "Category": md_category, "MD Confidence": det['conf'], "Common Name": common_name, "Scientific Name": scientific_name})
                    annotated_detections.append({"bbox": det['bbox'], "label": display_label})
        annotated_img = draw_annotations_on_pil(img_pil.copy(), annotated_detections)
        return annotated_img, pd.DataFrame(detection_data)
    except Exception as e:
        st.error(f"Error during SpeciesNet processing: {e}")
        if os.path.exists(temp_image_path): os.remove(temp_image_path)
        return img_pil, pd.DataFrame()

def process_frame_with_yolo_only(img_pil, yolo_model, conf, is_segmentation_model, iou_threshold_val, is_webcam_mode=False):
    img_array_rgb = np.array(img_pil)
    results = yolo_model.predict(img_array_rgb, conf=conf, iou=iou_threshold_val, verbose=False)
    detection_data = []
    class_id_counters = defaultdict(int)
    temp_annotated_detections = []
    annotated_img = img_pil.copy(); segmented_img = img_pil.copy()
    if results and len(results) > 0:
        res = results[0]
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                yolo_confidence = float(box.conf)
                class_id = int(box.cls); class_name = yolo_model.names[class_id]
                unique_id_for_frame = 'N/A'
                if is_webcam_mode:
                    class_id_counters[class_name] += 1
                    unique_id_for_frame = class_id_counters[class_name]
                display_label = f"{class_name} ID:{unique_id_for_frame} ({yolo_confidence:.2f})" if is_webcam_mode else f"{class_name} ({yolo_confidence:.2f})"
                detection_data.append({"Detector": "YOLOv11", "Class": class_name, "Confidence": yolo_confidence, "BBox (xyxy)": [x1, y1, x2, y2], "ID": unique_id_for_frame})
                img_width, img_height = img_pil.size
                temp_annotated_detections.append({"bbox": [x1/img_width, y1/img_height, (x2-x1)/img_width, (y2-y1)/img_height], "label": display_label})
        annotated_img = draw_annotations_on_pil(img_pil.copy(), temp_annotated_detections)
        if is_segmentation_model and res.masks is not None: segmented_img = Image.fromarray(res.plot())
    return annotated_img, segmented_img, pd.DataFrame(detection_data)

def process_frame_with_resnet(img_pil, yolo_model, resnet_model, resnet_transforms, imagenet_classes, conf, iou_threshold_val):
    img_array_rgb = np.array(img_pil)
    results = yolo_model.predict(img_array_rgb, conf=conf, iou=iou_threshold_val, verbose=False)
    detection_data, annotated_detections = [], []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            yolo_confidence = float(box.conf)
            cropped_image_pil = img_pil.crop((x1, y1, x2, y2))
            classified_species, classification_confidence = get_species_from_resnet(cropped_image_pil, resnet_model, resnet_transforms, imagenet_classes)
            detection_data.append({"Detector": "YOLOv11", "YOLO Confidence": yolo_confidence, "Species": classified_species, "ResNet Confidence": classification_confidence, "BBox (xyxy)": [x1, y1, x2, y2]})
            img_width, img_height = img_pil.size
            annotated_detections.append({"bbox": [x1/img_width, y1/img_height, (x2-x1)/img_width, (y2-y1)/img_height], "label": f"{classified_species} ({classification_confidence:.2f})"})
    annotated_img = draw_annotations_on_pil(img_pil.copy(), annotated_detections)
    return annotated_img, pd.DataFrame(detection_data)

# ... (DeepSORT / ByteTrack helpers are same as before) ...
class DeepSORT_Track:
    _next_id = 1
    def __init__(self, bbox, class_name, confidence):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7); self.kf.F[0,4]=1; self.kf.F[1,5]=1; self.kf.F[2,6]=1
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10.; self.kf.P[4:, 4:] *= 1000.; self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01; self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_kf_state(bbox)
        self.time_since_update = 0; self.id = DeepSORT_Track._next_id; DeepSORT_Track._next_id += 1
        self.class_name = class_name; self.confidence = confidence; self.hits = 0; self.age = 0; self.max_age = 50 
    def _convert_bbox_to_kf_state(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w/2, y + h/2, w/h, h]).reshape((4, 1))
    def get_bbox_from_kf_state(self):
        x_center, y_center, aspect_ratio, h = self.kf.x[:4].flatten()
        w = aspect_ratio * h
        return [x_center - w/2, y_center - h/2, w, h]
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0: self.kf.x[6] *= 0.0
        self.kf.predict(); self.age += 1; self.time_since_update += 1
    def update(self, bbox, class_name, confidence):
        self.time_since_update = 0; self.hits += 1; self.class_name = class_name; self.confidence = confidence
        self.kf.update(self._convert_bbox_to_kf_state(bbox))
    def get_state(self): return self.get_bbox_from_kf_state()

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1; x2, y2, w2, h2 = bbox2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap; union_area = w1 * h1 + w2 * h2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    if not tracks: return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if not detections: return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(tracks))
    cost_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, track in enumerate(tracks): cost_matrix[d, t] = 1 - iou(det["bbox"], track.get_state())
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_detections, unmatched_tracks = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < (1 - iou_threshold): matches.append([r, c])
        else: unmatched_detections.append(r); unmatched_tracks.append(c)
    for d in range(len(detections)): 
        if d not in row_ind: unmatched_detections.append(d)
    for t in range(len(tracks)):
        if t not in col_ind: unmatched_tracks.append(t)
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)

# --- REPLACED: VIDEO PROCESSING FUNCTION (Colab Style - Fixes Blank Video on Windows) ---
def process_video_frames(uploaded_video_file, yolo_model, conf, is_segmentation_model, iou_threshold_val, tracker_type="DeepSORT"):
    temp_input_video_path = save_uploaded_file(uploaded_video_file)
    if not temp_input_video_path: return None, None, pd.DataFrame(), {}
    
    cap = cv2.VideoCapture(temp_input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use tempfile for raw output
    output_annotated_video_path_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated_raw.mp4").name
    output_segmented_video_path_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_segmented_raw.mp4").name
    
    # Force 'avc1' (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out_annotated = cv2.VideoWriter(output_annotated_video_path_raw, fourcc, fps, (width, height))
    out_segmented = cv2.VideoWriter(output_segmented_video_path_raw, fourcc, fps, (width, height))
    
    species_frames = {}
    active_deepsort_tracks = []
    DeepSORT_Track._next_id = 1
    session_unique_species_ids = defaultdict(lambda: {'next_id': 1, 'assigned_ids': {}})
    
    progress_bar = st.progress(0, "Processing video..."); status_text = st.empty(); frame_num = 0
    
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        current_frame_detections_for_counting = []
        segmented_frame_pil = frame_pil.copy()
        
        if tracker_type == "ByteTrack":
            results = yolo_model.track(frame_rgb, persist=True, tracker="bytetrack.yaml", conf=conf, iou=iou_threshold_val, verbose=False)
            if results:
                res = results[0]
                if is_segmentation_model and res.masks:
                     segmented_frame_pil = Image.fromarray(res.plot())
                
                if res.boxes and res.boxes.id is not None:
                    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = res.boxes.id.cpu().numpy().astype(int)
                    clss = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    
                    for box, track_id, cls, conf_val in zip(boxes, track_ids, clss, confs):
                        class_name = yolo_model.names[cls]
                        x1, y1, x2, y2 = box
                        current_frame_detections_for_counting.append({'bbox': [x1, y1, x2-x1, y2-y1], 'class': class_name, 'conf': float(conf_val), 'id': int(track_id)})
                        if class_name.lower() in TARGET_ANIMAL_CLASSES:
                             if class_name not in species_frames:
                                 species_frames[class_name] = segmented_frame_pil.copy() if is_segmentation_model else frame_pil.copy()
        else: # DeepSORT
            _, segmented_frame_pil, frame_detections_df_raw = process_frame_with_yolo_only(frame_pil, yolo_model, conf, is_segmentation_model, iou_threshold_val)
            current_detections_for_tracker = []
            for _, row in frame_detections_df_raw.iterrows():
                x1, y1, x2, y2 = row['BBox (xyxy)']
                current_detections_for_tracker.append({"bbox": [x1, y1, x2-x1, y2-y1], "class_name": row['Class'], "confidence": row['Confidence']})
                if row['Class'].lower() in TARGET_ANIMAL_CLASSES:
                     if row['Class'] not in species_frames:
                         species_frames[row['Class']] = segmented_frame_pil.copy() if is_segmentation_model else frame_pil.copy()
            
            for t in active_deepsort_tracks: t.predict()
            matches, unmatched_detections, _ = associate_detections_to_tracks(current_detections_for_tracker, active_deepsort_tracks, iou_threshold=0.5)
            for d_idx, t_idx in matches: active_deepsort_tracks[t_idx].update(current_detections_for_tracker[d_idx]["bbox"], current_detections_for_tracker[d_idx]["class_name"], current_detections_for_tracker[d_idx]["confidence"])
            for d_idx in unmatched_detections:
                det = current_detections_for_tracker[d_idx]
                if det['class_name'].lower() in TARGET_ANIMAL_CLASSES:
                    active_deepsort_tracks.append(DeepSORT_Track(det["bbox"], det["class_name"], det["confidence"]))
            
            active_deepsort_tracks = [t for t in active_deepsort_tracks if t.time_since_update < t.max_age]
            for track in active_deepsort_tracks:
                if track.class_name.lower() in TARGET_ANIMAL_CLASSES:
                    x, y, w, h = track.get_state()
                    current_frame_detections_for_counting.append({'bbox': [x, y, w, h], 'class': track.class_name, 'conf': track.confidence, 'id': track.id})

        annotated_img_with_tracking = frame_pil.copy()
        draw = ImageDraw.Draw(annotated_img_with_tracking)
        tracked_detections_for_draw = []
        
        # --- FIX: ACTIVE COUNTING FOR OVERLAY TEXT ---
        active_counts_in_frame = defaultdict(int)

        for obj in current_frame_detections_for_counting:
            cls_name = obj['class']
            track_id = obj['id']
            species_id_map = session_unique_species_ids[cls_name]['assigned_ids']
            if track_id not in species_id_map:
                species_id_map[track_id] = session_unique_species_ids[cls_name]['next_id']
                session_unique_species_ids[cls_name]['next_id'] += 1
            
            # Count for this frame
            active_counts_in_frame[cls_name] += 1
            
            x, y, w, h = obj['bbox']
            label = f"{cls_name} ID:{species_id_map[track_id]} ({obj['conf']:.2f})"
            img_w, img_h = annotated_img_with_tracking.size
            tracked_detections_for_draw.append({"bbox": [x/img_w, y/img_h, w/img_w, h/img_h], "label": label})
        
        annotated_img_with_tracking = draw_annotations_on_pil(annotated_img_with_tracking, tracked_detections_for_draw)
        
        # Display ACTIVE counts, not Total
        count_text = "\n".join([f"{k}: {v}" for k, v in active_counts_in_frame.items()])
        if count_text:
            try: font = ImageFont.truetype("arial.ttf", 20)
            except: font = ImageFont.load_default()
            draw.text((10, 10), count_text, font=font, fill=(0, 255, 0))
        
        out_annotated.write(cv2.cvtColor(np.array(annotated_img_with_tracking), cv2.COLOR_RGB2BGR))
        out_segmented.write(cv2.cvtColor(np.array(segmented_frame_pil), cv2.COLOR_RGB2BGR))
        
        frame_num += 1
        progress = min(1.0, frame_num / total_frames)
        progress_bar.progress(progress)
        status_text.text(f"Processed frame {frame_num}/{total_frames}")
        
    cap.release()
    out_annotated.release()
    out_segmented.release()
    os.remove(temp_input_video_path)
    progress_bar.empty()
    status_text.empty()
    
    # FFmpeg Compression (Critical for Browser Playback)
    output_annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4").name
    output_segmented_video_path = tempfile.NamedTemporaryFile(delete=False, suffix="_segmented.mp4").name
    
    def compress_video(inp, out):
        try:
            # Force H.264 compression via FFmpeg
            subprocess.run(['ffmpeg', '-i', inp, '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast', '-y', out], capture_output=True)
            if os.path.exists(inp): os.remove(inp)
            return out
        except Exception: 
            return inp # Return raw if ffmpeg fails (will show blank on Windows if raw is mp4v)
        
    annotated_final = compress_video(output_annotated_video_path_raw, output_annotated_video_path)
    segmented_final = compress_video(output_segmented_video_path_raw, output_segmented_video_path)
    
    summary_data = [{"Class": k, "Total Count": len(v['assigned_ids'])} for k, v in session_unique_species_ids.items() if v['assigned_ids']]
    return annotated_final, segmented_final, pd.DataFrame(summary_data), species_frames

class YOLOVideoProcessor(VideoTransformerBase):
    def __init__(self, model_name, conf, iou, alert_enabled_dict, alert_manager, serial_conn, tracker_type, iot_on):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.alert_enabled_dict = alert_enabled_dict
        self.alert_manager = alert_manager
        self.serial_conn = serial_conn 
        self.iot_on = iot_on
        
        # Determine Tracker Config based on selection
        if tracker_type == "ByteTrack":
            self.tracker_file = "bytetrack.yaml"
        else:
            self.tracker_file = "botsort.yaml" # Ultralytics implementation of DeepSORT/BoT-SORT

        # State for population counting (Class -> {TrackingID: UniqueSessionID})
        self.session_ids = defaultdict(lambda: {'next': 1, 'map': {}})

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # USE TRACKING instead of simple prediction
        # persist=True is key for ID consistency across frames
        results = self.model.track(img, persist=True, tracker=self.tracker_file, conf=self.conf, iou=self.iou, verbose=False)
        
        annotated_img = img.copy()
        
        if results:
            res = results[0]
            # Use plot() but we will draw over it or use it as base
            annotated_img = res.plot() 
            
            if res.boxes and res.boxes.id is not None:
                # Extract tracking data
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)
                
                detected_counts = defaultdict(int)

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    class_name = self.model.names[int(cls)]
                    
                    # 1. ASSIGN UNIQUE SESSION ID
                    # This keeps the ID simple (1, 2, 3) instead of (45, 46, 47)
                    if track_id not in self.session_ids[class_name]['map']:
                        self.session_ids[class_name]['map'][track_id] = self.session_ids[class_name]['next']
                        self.session_ids[class_name]['next'] += 1
                    
                    unique_id = self.session_ids[class_name]['map'][track_id]
                    detected_counts[class_name] += 1
                    
                    # 2. DRAW ID ON FRAME (Overlay on top of plot)
                    # We draw a small extra label to ensure ID is visible
                    x1, y1, x2, y2 = box
                    label = f"ID: {unique_id}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
                    cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # 3. ALERT LOGIC (Strict Filter)
                    if class_name.lower() in TARGET_ANIMAL_CLASSES:
                        if self.alert_manager.should_alert(class_name):
                            msg = f"Alert: {class_name} detected via Webcam (ID: {unique_id})"
                            print(f"🚀 ALERT TRIGGERED: {msg}")
                            
                            # Digital Alerts
                            if self.alert_enabled_dict['sms']: threading.Thread(target=send_sms_alert, args=(msg,)).start()
                            if self.alert_enabled_dict['telegram']: 
                                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                threading.Thread(target=send_telegram_alert, args=(msg, {class_name: pil_img})).start()
                            if self.alert_enabled_dict['email']: threading.Thread(target=send_email_alert, args=(msg,)).start()
                            if self.alert_enabled_dict['call']: threading.Thread(target=send_voice_call_alert, args=(msg,)).start()
                            
                            # IoT Alert
                            if self.iot_on: 
                                threading.Thread(target=send_iot_alert_direct).start()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --------------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------------

if mode == "Image":
    if image_classifier_choice == "SpeciesNet":
        speciesnet_model = load_speciesnet_model()
        if speciesnet_model:
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                col1, col2 = st.columns(2)
                with col1: st.image(uploaded_image, caption="Input", use_container_width=True)
                with st.spinner("Classifying..."):
                    annotated_img, summary_df = process_frame_with_speciesnet(Image.open(uploaded_image).convert("RGB"), speciesnet_model, conf)
                with col2: st.image(annotated_img, caption="Annotated", use_container_width=True)
                if not summary_df.empty:
                    st.dataframe(summary_df)
                    
                    # --- GOOGLE GEMINI INTEGRATION & CHATBOT ---
                    try:
                        # Extract the best animal match
                        if "Common Name" in summary_df.columns:
                            best_animal = summary_df.iloc[0]["Common Name"]
                        elif "SpeciesNet Prediction" in summary_df.columns:
                            best_animal = summary_df.iloc[0]["SpeciesNet Prediction"]
                        else:
                            best_animal = None

                        if best_animal and best_animal.lower() != "unknown species":
                            st.subheader(f"🧠 AI Insights: {best_animal}")
                            with st.spinner(f"Asking Google AI about {best_animal}..."):
                                info = generate_animal_info(best_animal)
                                st.markdown(info)

                            # --- CHATBOT SECTION ---
                            st.divider()
                            st.subheader(f"💬 Ask about the {best_animal}")
                            user_q = st.text_input(f"Do you have any specific question about {best_animal}?", key="chat_sn")
                            if user_q:
                                with st.spinner("Analyzing question..."):
                                    chat_ans = ask_gemini_chatbot(best_animal, user_q)
                                    st.markdown(f"**AI Answer:** {chat_ans}")
                            # -----------------------

                    except Exception as e:
                        print(f"Gemini Error: {e}")
                    # ---------------------------------

                    # Alerts Removed for Image Mode
    
    elif image_classifier_choice == "ResNet50":
        yolo_model, _ = load_yolo_model(model_selection_image_resnet)
        resnet_model, resnet_transforms = load_resnet_model()
        imagenet_classes = get_imagenet_classes()
        if yolo_model and resnet_model:
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                col1, col2 = st.columns(2)
                with col1: st.image(uploaded_image, caption="Input", use_container_width=True)
                with st.spinner("Detecting..."):
                    annotated_img, summary_df = process_frame_with_resnet(Image.open(uploaded_image).convert("RGB"), yolo_model, resnet_model, resnet_transforms, imagenet_classes, conf, iou_threshold)
                with col2: st.image(annotated_img, caption="Annotated", use_container_width=True)
                if not summary_df.empty:
                    st.dataframe(summary_df)
                    
                    # --- GOOGLE GEMINI INTEGRATION & CHATBOT ---
                    try:
                        if "Species" in summary_df.columns:
                            best_animal = summary_df.iloc[0]["Species"]
                            if best_animal:
                                st.subheader(f"🧠 AI Insights: {best_animal}")
                                with st.spinner(f"Asking Google AI about {best_animal}..."):
                                    info = generate_animal_info(best_animal)
                                    st.markdown(info)
                                
                                # --- CHATBOT SECTION ---
                                st.divider()
                                st.subheader(f"💬 Ask about the {best_animal}")
                                user_q = st.text_input(f"Do you have any specific question about {best_animal}?", key="chat_rn")
                                if user_q:
                                    with st.spinner("Analyzing question..."):
                                        chat_ans = ask_gemini_chatbot(best_animal, user_q)
                                        st.markdown(f"**AI Answer:** {chat_ans}")
                                # -----------------------

                    except Exception as e:
                        print(f"Gemini Error: {e}")
                    
                    # Alerts Removed for Image Mode

elif mode == "Video":
    yolo_model, is_segmentation_model = load_yolo_model(model_selection_video_webcam)
    if yolo_model:
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            # 1. DISPLAY INPUT VIDEO FIRST
            st.subheader("Input Video")
            st.video(uploaded_video)

            if st.button("Process Video for Detections & Segmentation"):
                annotated_path, segmented_path, summary_df, species_frames = process_video_frames(uploaded_video, yolo_model, conf, is_segmentation_model, iou_threshold, tracker_type)
                if annotated_path:
                    st.success("Processing complete!")
                    st.dataframe(summary_df)

                    # 2. DISPLAY OUTPUTS SIDE-BY-SIDE (HTML Embed Style)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Annotated Video")
                        with open(annotated_path, 'rb') as f:
                            video_bytes = f.read()
                            video_b64 = b64encode(video_bytes).decode()
                            # HTML Embed (Fixes Blank Video Issue)
                            html(f'''<video width="100%" controls autoplay muted loop>
                                     <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                                     </video>''', height=350)
                        
                        with open(annotated_path, 'rb') as f:
                            st.download_button("Download Annotated", f.read(), file_name="annotated.mp4")

                    with col2:
                        if is_segmentation_model:
                            st.subheader("Segmented Video")
                            with open(segmented_path, 'rb') as f:
                                video_bytes = f.read()
                                video_b64 = b64encode(video_bytes).decode()
                                html(f'''<video width="100%" controls autoplay muted loop>
                                         <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                                         </video>''', height=350)

                            with open(segmented_path, 'rb') as f:
                                st.download_button("Download Segmented", f.read(), file_name="segmented.mp4")
                        else:
                             st.info("Segmentation model not selected.")
                    
                    summary_string = get_alert_summary(summary_df)
                    if _should_send_alert(summary_df, mode):
                        if sms_enabled: send_sms_alert(summary_string)
                        if call_enabled: send_voice_call_alert(summary_string)
                        if email_enabled: send_email_alert(summary_string)
                        if telegram_enabled: send_telegram_alert(summary_string, species_frames)
                        
                        # --- IOT TRIGGER (VIDEO) ---
                        if iot_enabled: 
                             threading.Thread(target=send_iot_alert_direct).start()

elif mode == "Webcam":
    st.header("Real-Time Autonomous Detection")
    current_alert_manager = st.session_state.get('alert_manager', AlertManager())
    # Add 'iot' to the settings dictionary passed to processor
    alert_settings = {
        'sms': sms_enabled,
        'call': call_enabled,
        'email': email_enabled,
        'telegram': telegram_enabled,
        'iot': iot_enabled
    }
    
    # PASS THE SERIAL CONNECTION EXPLICITLY HERE
    conn = st.session_state.get('serial_conn')
    
    # --- FORCE GOOGLE STUN TO SAVE CREDITS ---
    google_stun_server = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    ctx = webrtc_streamer(
        key="wildlife-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=google_stun_server, # <--- Credit Saver
        video_processor_factory=lambda: YOLOVideoProcessor(model_selection_video_webcam, conf, 0.7, alert_settings, current_alert_manager, conn, tracker_type, iot_enabled),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Fixed indentation and variable check for ctx
    if ctx.state.playing: 
        st.write("Monitoring Active...")