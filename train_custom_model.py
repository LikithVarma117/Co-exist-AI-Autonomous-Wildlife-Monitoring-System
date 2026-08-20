from roboflow import Roboflow
from ultralytics import YOLO
import torch
import os
import shutil

# --- FIX MEMORY FRAGMENTATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train():
    # 1. Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ GPU not detected. Training will be slow.")
        device = 'cpu'

    # 2. Download Dataset to a SHORT PATH
    # We use a NEW folder to avoid mixing with previous data
    target_folder = "C:\\wildlife_data_v2"
    
    if os.path.exists(target_folder):
        print(f"🧹 Cleaning up old data at {target_folder}...")
        try:
            shutil.rmtree(target_folder)
        except:
            pass

    print(f"⬇️ Downloading Dataset Version 2 to {target_folder}...")
    
    rf = Roboflow(api_key="ivWOBAvfpuBCwAqxG29e")
    project = rf.workspace("species-zkfzi").project("coco-9")
    
    # --- CHANGED VERSION TO 2 ---
    version = project.version(2)
    dataset = version.download("yolov11", location=target_folder)
    
    data_yaml_path = os.path.join(target_folder, "data.yaml")
    print(f"📂 Config found at: {data_yaml_path}")

    # 3. Load Model
    print("🧠 Loading YOLOv11s...")
    model = YOLO('yolo11s.pt') 

    # 4. Train
    print("🚀 Starting Training...")
    try:
        model.train(
            data=data_yaml_path,
            epochs=20,
            imgsz=640,
            batch=4,        # Safe for 6GB VRAM
            device=device,
            workers=0,      # Windows Fix
            name='wildlife_coco9_v2', # New folder name for Version 2
            plots=True
        )
        print("🎉 Training Done! Best weights saved in 'runs/detect/wildlife_coco9_v2/weights/best.pt'")
    except Exception as e:
        print(f"❌ Training Failed: {e}")

if __name__ == '__main__':
    train()