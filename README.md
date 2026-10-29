# Garbage Detection using YOLO26

## What is YOLO26?
Ultralytics YOLO26 is a state-of-the-art computer vision model designed for high efficiency and real-time performance. It features a novel NMS-Free (Non-Maximum Suppression) architecture that simplifies the detection pipeline, making inference faster and more stable, especially on edge hardware. 

## Why YOLO26 for Garbage Detection?
YOLO26's streamlined architecture and MuSGD optimizer allow for rapid convergence during training and highly efficient inference. It provides excellent accuracy for object detection while keeping latencies low, making it ideal for real-time garbage detection where edge devices or limited hardware may be deployed.

## Dataset Organization
The dataset is configured in `unified_dataset/data.yaml` and should be organized as follows:
```text
unified_dataset/
├── data.yaml       # Configuration file with relative paths and classes
├── images/
│   ├── train/      # Training images (.jpg, .png, etc.)
│   └── val/        # Validation images
└── labels/
    ├── train/      # YOLO formatted text annotations for training
    └── val/        # YOLO formatted text annotations for validation
```
*(Note: At present, the `images/` directory is missing from the dataset folder. Please copy your images to the respective train and val folders before starting the training process.)*

## Installation & Dependencies
The project requires Python, PyTorch, and Ultralytics. The current virtual environment (`venv`) already has the necessary packages installed. To install or update dependencies manually:
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install ultralytics torch torchvision opencv-python
```

## How to Train the YOLO26 Model
Since the project currently lacks the `images/` directory, ensure the dataset is properly populated first. Then, run the training script:
```powershell
python train_yolo26.py --epochs 50 --batch 16
```
This script will:
- Download the pretrained `yolo26n.pt` base model.
- Fine-tune it on the Garbage Detection dataset.
- Save the best weights to `runs/detect/yolo26_garbage/weights/best.pt`.

## How to Validate the Model
To evaluate the trained YOLO26 model (or the original YOLOv8 model) on the validation split and get precision, recall, and mAP metrics, run:
```powershell
# Validate the newly trained YOLO26 model
python validate_yolo26.py

# Validate the original YOLOv8 model for comparison
python validate_yolo26.py --model best.pt
```

## How to Run Image and Video Predictions
The updated `predict.py` script automatically uses the newly trained YOLO26 model if it exists; otherwise, it falls back to the original `best.pt`. 

Run the prediction script and it will prompt you for an image/video path if not provided:
```powershell
# Run with prompted input
python predict.py

# Run on a specific image and save the result
python predict.py --source test.jpg --save

# Run using the original YOLOv8 model for comparison
python predict.py --model best.pt --source test.jpg --save

# Run webcam inference (if a camera is connected)
python predict.py --source 0
```

## Output Locations
- **Trained Model Checkpoint**: `runs/detect/yolo26_garbage/weights/best.pt`
- **Validation Results**: Saved under `runs/detect/yolo_val/`
- **Prediction Outputs**: Saved under `runs/detect/predict/` (when using `--save`)
