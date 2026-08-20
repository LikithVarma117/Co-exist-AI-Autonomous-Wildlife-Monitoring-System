from roboflow import Roboflow
from ultralytics import YOLO
import torch
import pandas as pd
import time
import os

# --- CONFIGURATION ---
# API KEY (Keep yours private in real apps)
ROBOFLOW_API_KEY = "mHiAqmyDSRzqpbNThRmz" 
PROJECT_NAME = "wildlife-detection-using-ai"
PROJECT_VERSION = 1

# Models to compare (Nano, Small, Medium)
MODELS_TO_TEST = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']

# Training settings (Optimized for RTX 3050)
EPOCHS = 10       # Higher = Better Accuracy. Keep 10 for a quick benchmark test.
BATCH_SIZE = 16   # Safe for 4GB/8GB VRAM
IMG_SIZE = 640

def main():
    # 1. Download Dataset
    print("⬇️ Downloading Dataset...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(PROJECT_NAME).project("coco-wildlife-wild-life-detection")
    version = project.version(PROJECT_VERSION)
    dataset = version.download("yolov11")
    data_yaml = os.path.join(dataset.location, "data.yaml")

    # List to store results for the table
    benchmark_data = []

    print(f"\n🚀 Starting Benchmark on {torch.cuda.get_device_name(0)}...")

    for model_name in MODELS_TO_TEST:
        print(f"\n-------------------------------------------------")
        print(f"🔄 Processing Model: {model_name}")
        print(f"-------------------------------------------------")

        # Load Model
        model = YOLO(model_name)

        # A. TRAIN (To get Accuracy Metrics)
        # We train briefly to see how well this architecture learns your specific data
        results = model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,
            verbose=False,
            plots=False,
            name=f"bench_{model_name.replace('.pt', '')}"
        )

        # Extract Metrics from the final training epoch
        # Ultralytics stores these in the 'results' object
        precision = results.box.mp   # Mean Precision
        recall = results.box.mr      # Mean Recall
        map50 = results.box.map50    # mAP @ 0.5
        map50_95 = results.box.map   # mAP @ 0.5:0.95

        # B. TEST SPEED (To get FPS)
        print(f"⚡ Testing Inference Speed for {model_name}...")
        # Run dummy inference to warm up
        model.predict(source=os.path.join(dataset.location, "valid/images"), max_det=1, device=0, verbose=False)
        
        start_time = time.time()
        # Predict on validation set (simulate video stream)
        model.predict(source=os.path.join(dataset.location, "valid/images"), device=0, verbose=False)
        end_time = time.time()
        
        # Calculate FPS based on number of images in valid set
        num_images = len(os.listdir(os.path.join(dataset.location, "valid/images")))
        total_time = end_time - start_time
        fps = num_images / total_time

        # Store Data
        benchmark_data.append({
            "Model Variant": model_name,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "mAP@0.50": round(map50, 3),
            "mAP@0.5:0.95": round(map50_95, 3),
            "FPS (GPU)": int(fps)
        })

    # --- GENERATE REPORT ---
    print("\n\n" + "="*60)
    print("FINAL EVALUATION TABLE (For Project Report)")
    print("="*60)
    
    df = pd.DataFrame(benchmark_data)
    
    # Display table in console
    print(df.to_string(index=False))
    
    # Save to CSV for Excel
    df.to_csv("model_evaluation_metrics.csv", index=False)
    print("\n✅ metrics saved to 'model_evaluation_metrics.csv'")

if __name__ == '__main__':
    main()