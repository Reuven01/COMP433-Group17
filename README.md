# COMP433-Group17 - Blood Cell Object Detection with YOLOv11

## 📘 Project Overview
This project implements an automated pipeline for **detecting and classifying blood cells** in microscope smear images using **YOLOv11**, a modern one-stage object detection model.

The model performs:

- 🔍 Bounding box detection  
- 🧬 Cell classification (RBC, WBC, Platelet)  

Our approach initially explored SAM2-based segmentation, but given the nature of available datasets (bounding-box annotations), we shifted to a single-model YOLOv11 pipeline for simplicity, efficiency, and improved performance.

---

## 📁 Repository Structure

Everything can be run via the google Colab file (given that you have the Key provided in the Project Google Drive Folder)

---

## 🧪 Requirements

This project was primarily developed on **Google Colab**, which already includes most required libraries.

Minimal dependencies:

ultralytics
torch
numpy
pandas

If running locally:

```bash
pip install ultralytics torch numpy pandas
```

⸻

📥 Dataset Access

The full training dataset used in this project was created by aggregating several Roboflow datasets inside our personal Roboflow workspace, where we also applied augmentations (e.g., vertical flips, color/brightness adjustments).
Because of this, the combined dataset cannot be publicly downloaded as a single file.

🔑 Access via Roboflow API Key

To load the dataset:
	1.	Obtain the ROBOFLOW_API_KEY from the Google Drive project folder.
	2.	Add it as a Google Colab secret:
	•	In Colab: Settings → Secrets → Add new secret
	•	Name: ROBOFLOW_API_KEY
	•	Value: (provided in Drive)
	3.	Training scripts will automatically access the dataset using this key.
⸻

🔍 Running Inference

After training the block of code below can be ran
```
yolo detect predict \
    model=models/best.pt \
    source=sample_test/ \
    imgsz=640 \
    save=True
```
Results will appear in:

`runs/detect/predict/`

⸻

📊 Evaluation
```
yolo detect val \
    model=models/best.pt \
    data=datasets/data.yaml \
    imgsz=640
```
Metrics include:
	•	mAP@50 and mAP@50–95
	•	Precision & recall
	•	Confusion matrix
	•	PR and F1 curves

Evaluation outputs may be saved in a results/ directory.

⸻

🧠 Implementation Notes
- Training and inference were performed using Google Colab GPU.
- YOLOv11 combines detection and classification in one efficient model.
- Platelet-rich datasets greatly improved performance on extremely small objects.
- Entire model is built using PyTorch through the Ultralytics YOLO framework.

⸻

👥 Team
- Justin Sciortino - 40247931
- Gabriel Derhy - 40247254
- Carlo Ramadori - 40243639
- Reuven Minciotti - 40252872

COMP 433 – Introduction to Deep Learning
Fall 2025
