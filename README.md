# COMP433-Group17 - Blood Cell Object Detection with YOLOv11

## 📘 Project Overview

This project trains and evaluates **YOLOv11** models for **detecting and classifying blood cells** in microscope smear images, a modern one-stage object detection model. The system performs bounding box object detection for three main blood cell types: **Red Blood Cells (RBC)**, **White Blood Cells (WBC)**, and **Platelets**.

### Key Features

- 🔍 **Bounding box detection** for blood cell components
- 🧬 **Multi-class classification** (RBC, WBC, Platelet)
- 🚀 **Automated training pipeline** with hyperparameter experimentation
- 📊 **Comprehensive evaluation** with mAP metrics and visualizations
- 🔄 **Resume capability** for interrupted training sessions

### Project Approach

Our approach initially explored segmentation, but given the nature of available datasets (bounding-box annotations), we shifted to YOLOv11 for simplicity, efficiency, and improved performance. The project systematically experiments with different model architectures (YOLOv11 nano and small variants) and batch sizes to find optimal configurations for blood cell detection.

---

## 📁 Repository Structure

### Main Files

- **`comp433_sota_project_group_17.py`** - Main training script for local execution
  - Trains YOLOv11 models (nano, small, and medium variants) with different batch sizes
  - Automatically downloads dataset from Roboflow using API key
  - Saves experiment results to CSV for comparison
  - Supports checkpoint resuming for long training sessions

- **`COMP433_SOTA_Project_Group_17.ipynb`** - Google Colab notebook version
  - Similar functionality to the Python script
  - Includes Colab-specific setup (GPU checking, Drive mounting)
  - Designed for cloud-based training with GPU acceleration
  - Also includes YOLOv11 medium variant in training experiments

> **Note**: The `.ipynb` and `.py` files are functionally similar. The notebook was used earlier in the project for Colab-based training and provides better readability for interactive development, while the Python script is optimized for execution.

### Helper Scripts

- **`checkenv.py`** - Environment verification utility
  - Checks if `ROBOFLOW_API_KEY` environment variable is properly set
  - Useful for debugging API key configuration issues

- **`infer_all_best_models.py`** - Batch inference script
  - Automatically finds all trained models (`best.pt` files) in the `runs/` directory
  - Runs inference on all models for comparison
  - Processes images from the `inference/` directory
  - Saves annotated results organized by model

### Directory Structure

```
COMP433_Project/
├── comp433_sota_project_group_17.py    # Main training script
├── COMP433_SOTA_Project_Group_17.ipynb  # Colab notebook
├── checkenv.py                          # Environment checker
├── infer_all_best_models.py             # Batch inference tool
├── datasets/                             # Dataset directory (downloaded from Roboflow)
│   └── Blood-Smear-Components---yolov11-2/
├── runs/                                 # Training outputs and model weights
│   └── [experiment_name]/
│       └── weights/
│           ├── best.pt                   # Best model checkpoint
│           └── last.pt                   # Latest checkpoint
├── inference/                            # Test images for inference
└── yolo11_experiment_results.csv        # Training results summary
```

---

## 🧪 Requirements

### Dependencies

- `ultralytics` - YOLOv11 framework
- `torch` - PyTorch deep learning library
- `numpy` - Numerical computing
- `pandas` - Data manipulation and CSV handling
- `roboflow` - Dataset access via API
- `python-dotenv` - Environment variable management
- `PIL` (Pillow) - Image processing

### Installation

For local execution:

```bash
pip install ultralytics torch numpy pandas roboflow python-dotenv pillow
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Environment Setup

1. Create a `.env` file in the project root:
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

2. Verify environment setup:
   ```bash
   python checkenv.py
   ```

---

## 🚀 Running the Project

### Option 1: Google Colab (Recommended for Training)

1. **Upload the notebook** `COMP433_SOTA_Project_Group_17.ipynb` to Google Colab

2. **Set up Roboflow API Key**:
   - In Colab: Settings → Secrets → Add new secret
   - Name: `ROBOFLOW_API_KEY`
   - Value: (obtain from project Google Drive folder)

3. **Run the notebook cells**:
   - The notebook will automatically mount Google Drive
   - Download the dataset from Roboflow
   - Train models with GPU acceleration
   - Results are saved to your Drive

### Option 2: Local Execution

1. **Set up environment variables**:
   ```bash
   # Create .env file or export directly
   export ROBOFLOW_API_KEY=your_api_key_here
   ```

2. **Run training script**:
   ```bash
   python comp433_sota_project_group_17.py
   ```

   The script will:
   - Download the dataset from Roboflow (if not already present)
   - Train multiple model configurations (YOLOv11n and YOLOv11s with batch sizes 8, 16, 32, 64)
   - Save results to `yolo11_experiment_results.csv`
   - Resume from checkpoints if training is interrupted

3. **Monitor training progress**:
   - Check `runs/[experiment_name]/` for training outputs
   - View `yolo11_experiment_results.csv` for metrics comparison

### Training Configuration

The training script uses the following default configuration:

- **Models**: `yolo11n.pt` (nano), `yolo11s.pt` (small)
- **Epochs**: 450
- **Batch sizes**: [8, 16, 32, 64]
- **Optimizer**: Auto (automatically selected by Ultralytics)
- **Patience**: 100 epochs (early stopping)
- **Image size**: 640x640

---

## 🔍 Running Inference

### Single Model Inference

After training, run inference on a specific model:

```bash
yolo detect predict \
    model=runs/yolo11n_b16/weights/best.pt \
    source=inference/ \
    imgsz=640 \
    save=True
```

Results will appear in `runs/detect/predict/`

### Batch Inference (All Models)

Use the helper script to run inference on all trained models:

```bash
python infer_all_best_models.py \
    --runs-dir runs \
    --images-dir inference \
    --batch-size 8 \
    --out inference/results
```

This will:
- Find all `best.pt` files in the `runs/` directory
- Process all images in the `inference/` directory
- Save annotated results organized by model in `inference/results/`

### Custom Inference

```bash
python infer_all_best_models.py \
    --images path/to/image1.jpg path/to/image2.jpg \
    --out custom_output/
```

---

## 📊 Evaluation

### Validation Metrics

Run validation on a trained model:

```bash
yolo detect val \
    model=runs/yolo11n_b16/weights/best.pt \
    data=datasets/Blood-Smear-Components---yolov11-2/data.yaml \
    imgsz=640
```

### Metrics Included

- **mAP@50** - Mean Average Precision at IoU threshold 0.50
- **mAP@50-95** - Mean Average Precision averaged over IoU thresholds 0.50-0.95
- **Precision & Recall** - Per-class and overall metrics
- **Confusion Matrix** - Classification accuracy visualization
- **PR and F1 Curves** - Performance curves

### Results Analysis

- Training metrics are automatically saved to `yolo11_experiment_results.csv`
- Visual outputs (curves, confusion matrices) are in `runs/[experiment_name]/`
- Compare different configurations using the CSV file

---

## 📥 Dataset Access

The full training dataset was created by aggregating several Roboflow datasets in our personal Roboflow workspace, with applied augmentations (e.g., vertical flips, color/brightness adjustments). Because of this, the combined dataset cannot be publicly downloaded as a single file.

### Access via Roboflow API Key

1. Obtain the `ROBOFLOW_API_KEY` from the Google Drive project folder
2. Set it as an environment variable (see [Environment Setup](#environment-setup))
3. The training scripts will automatically download the dataset on first run

The dataset structure follows YOLOv11 format:
- `train/` - Training images and labels
- `valid/` - Validation images and labels
- `test/` - Test images and labels
- `data.yaml` - Dataset configuration file

---

## 🧠 Implementation Notes

- **Training**: Performed using Google Colab GPU (Tesla T4) for faster iteration
- **Framework**: Built entirely using PyTorch through the Ultralytics YOLO framework
- **Model Architecture**: YOLOv11 combines detection and classification in one efficient model
- **Small Object Detection**: Platelet-rich datasets greatly improved performance on extremely small objects
- **Checkpoint Management**: Automatic checkpoint saving and resuming prevents data loss during long training sessions
- **Experiment Tracking**: CSV-based results tracking enables easy comparison of different configurations

---

## 🔮 Future Vision

### Short-term Goals

1. **Enhanced Classification Granularity**
   - Extend detection to more specific blood cell components
   - Classify white blood cell subtypes (neutrophils, lymphocytes, monocytes, eosinophils, basophils)
   - Detect abnormal cell morphologies and rare cell types

2. **Improved Model Performance**
   - Experiment with larger YOLOv11 variants (medium, large, x-large)
   - Fine-tune hyperparameters for specific cell types
   - Implement ensemble methods for improved accuracy

### Long-term Vision: Pipeline Architecture

The primary long-term goal is to prove the usability of this robust detection model as the **foundation of a modular blood smear analysis pipeline**:

```
┌─────────────────────────────────────────┐
│  YOLOv11 Detection Model (This Project) │
│  - Elementary cell detection            │
│  - Basic classification (RBC/WBC/Plate)│
│  - Bounding box localization            │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Pipeline Router      │
    │  (Routes by cell type)│
    └──────┬───────────────┘
           │
    ┌──────┴──────┬──────────────┬─────────────┐
    │             │              │             │
    ▼             ▼              ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ WBC    │  │ RBC      │  │ Platelet │  │ Anomaly  │
│ Subtype│  │ Analysis │  │ Analysis │  │ Detection│
│ Model  │  │ Model    │  │ Model    │  │ Model    │
└────────┘  └──────────┘  └──────────┘  └──────────┘
```

**Key Benefits of This Architecture:**

1. **Modularity**: Each downstream task uses a specialized, lightweight model
2. **Efficiency**: Simple models can focus on specific tasks without redundant detection
3. **Scalability**: Easy to add new analysis modules (e.g., cell counting, morphology analysis)
4. **Maintainability**: Each component can be updated independently
5. **Robustness**: The primary detection model handles the complex task of finding and classifying cells, while downstream models perform simpler, focused analyses

**Downstream Applications:**

- **Anomaly Detection**: Lightweight models can identify abnormal cell morphologies, parasites, or disease markers
- **Cell Counting**: Automated counting of different cell types for diagnostic purposes
- **Morphology Analysis**: Detailed shape and size analysis for specific cell types
- **Disease Classification**: Specialized models for detecting malaria, leukemia, and other blood disorders
- **WBC Subtype Classification**: Identify specific white blood cell types (neutrophils, lymphocytes, etc.) from isolated WBC components

This pipeline approach allows the robust YOLOv11 model to handle the computationally intensive 
detection task, while downstream models can be simpler, faster, and more focused on their specific tasks.

---

## 👥 Team

- Justin Sciortino - 40247931
- Gabriel Derhy - 40247254
- Carlo Ramadori - 40243639
- Reuven Minciotti - 40252872

**COMP 433 – Introduction to Deep Learning**  
Fall 2025
