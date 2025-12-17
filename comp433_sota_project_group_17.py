## COMP433 - Blood Smear Analysis Object Detection
## This document demonstrates how to perform object detection on blood cell images using the YOLOv11 model with datasets from Roboflow. 
## It covers setting up the environment, downloading the dataset, training the model, and evaluating the results.

import os
import subprocess
import pandas as pd
import torch
from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO
from dotenv import load_dotenv

# Import ultralytics and run checks to verify the environment
ultralytics.checks()

# Load environment variables from .env file
load_dotenv()

# Set up Roboflow API key and download dataset
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# ====== SETUP ======
DRIVE_BASE = os.getcwd()
DATA_PATH = f"{DRIVE_BASE}/datasets/Blood-Smear-Components---yolov11-2/data.yaml"
RESULTS_CSV = os.path.join(DRIVE_BASE, "yolo11_experiment_results.csv")
RUNS_DIR = os.path.join(DRIVE_BASE, "runs")

os.makedirs(RUNS_DIR, exist_ok=True)

# ====== CONFIGURATION ======
model_names = ["yolo11n.pt", "yolo11s.pt"]
epochs = 450
batches = [8, 16, 32, 64]
optimizer = "auto"
patience = 100  # default value

# ====== LOAD EXISTING RESULTS IF ANY ======
if os.path.exists(RESULTS_CSV):
    print(f"📂 Found existing results file at {RESULTS_CSV}, resuming from it...")
    if os.path.getsize(RESULTS_CSV) > 0:
        try:
            results_table = pd.read_csv(RESULTS_CSV).to_dict("records")
            completed_experiments = {r["experiment"] for r in results_table}
        except pd.errors.EmptyDataError:
            print("⚠️ Results file malformed, starting fresh.")
            results_table = []
            completed_experiments = set()
    else:
        print("⚠️ Results file is empty, starting fresh.")
        results_table = []
        completed_experiments = set()
else:
    results_table = []
    completed_experiments = set()

# ====== MAIN TRAINING LOOP ======
for model_name in model_names:
    for batch in batches:
        exp_name = f"{model_name.split('.')[0]}_b{batch}"
        print(f"\n🚀 Starting experiment: {exp_name}")

        # Skip already completed experiments
        if exp_name in completed_experiments:
            print(f"⏩ Skipping {exp_name} (already completed)")
            continue

        # Path to expected checkpoint (Ultralytics structure)
        EXP_DIR = os.path.join(RUNS_DIR, exp_name)
        WEIGHTS_DIR = os.path.join(EXP_DIR, "weights")
        MODEL_CHECKPOINT = os.path.join(WEIGHTS_DIR, "last.pt")

        # ====== INITIALIZE MODEL ======
        if os.path.exists(MODEL_CHECKPOINT):
            print(f"🔄 Found checkpoint for {exp_name}, resuming training...")
            model = YOLO(MODEL_CHECKPOINT)
            resume_flag = True
        else:
            print(f"🚀 Starting new training for {exp_name}...")
            model = YOLO(model_name)
            resume_flag = False

        try:
            # ====== TRAIN ======
            results = model.train(
                data=DATA_PATH,
                epochs=epochs,
                batch=batch,
                patience=patience,
                optimizer=optimizer,
                name=exp_name,                  # experiment folder name
                project=RUNS_DIR,               # stored in /runs/
                exist_ok=True,
                resume=resume_flag,
                save=True,
                device=0,
                save_period=1
            )

            # ====== VALIDATE ======
            print(f"📊 Validating {exp_name} ...")
            val_results = model.val()

            # ====== SAVE RESULTS ======
            results_table.append({
                "experiment": exp_name,
                "model": model_name,
                "epochs": epochs,
                "batch": batch,
                "optimizer": optimizer,
                "mAP50": val_results.box.map50,
                "mAP50-95": val_results.box.map
            })

            pd.DataFrame(results_table).to_csv(RESULTS_CSV, index=False)
            print(f"💾 Saved results to {RESULTS_CSV}")

            # Free GPU memory
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Error during {exp_name}: {e}")
            pd.DataFrame(results_table).to_csv(RESULTS_CSV, index=False)
            print("💾 Progress saved before crash.")
            break

print("\n✅ All experiments complete! Final results saved to:")
print(RESULTS_CSV)