"""
train_yolo26_latest.py
Training script for YOLO26 Garbage Detection baseline.
Uses the Ultralytics Python API to train the official YOLO26 Nano model.
Preserves existing project checkpoints (best.pt) by isolating runs in 'runs/garbage_detection'.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 for Garbage Detection")
    parser.add_argument("--data", type=str, default="datasets/garbage_yolo26/data.yaml",
                        help="Path to dataset data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                        help="Pretrained YOLO26 checkpoint (e.g. yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train (e.g., 100 for full training, 1-3 for CPU quick baseline)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Dataloader workers (default: 2)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use: '0', 'cpu', or leave empty for auto-detection")
    parser.add_argument("--project", type=str, default="runs/garbage_detection",
                        help="Directory to save experiment runs")
    parser.add_argument("--name", type=str, default="yolo26n_baseline",
                        help="Experiment name")
    parser.add_argument("--exist_ok", action="store_true", default=False,
                        help="Overwrite existing run directory if present")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of dataset to train on (default: 1.0 for full dataset)")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_yaml_path = project_root / args.data if not os.path.isabs(args.data) else Path(args.data)
    model_path = project_root / args.model if not os.path.isabs(args.model) else Path(args.model)

    print("=================================================================")
    print("           YOLO26 GARBAGE DETECTION TRAINING PIPELINE            ")
    print("=================================================================")
    print(f"Project Directory : {project_root}")
    print(f"Dataset YAML      : {data_yaml_path}")
    print(f"Base Model        : {model_path}")
    print(f"Target Epochs     : {args.epochs}")
    print(f"Batch Size        : {args.batch}")
    print(f"Image Size        : {args.imgsz}")
    print(f"Workers           : {args.workers}")
    print(f"Dataset Fraction  : {args.fraction}")

    # 1. Dataset Verification
    if not data_yaml_path.exists():
        print(f"ERROR: Dataset YAML not found at: {data_yaml_path}", file=sys.stderr)
        sys.exit(1)

    print("\n[Step 1] Validating dataset integrity...")
    try:
        dataset_info = check_det_dataset(str(data_yaml_path))
        print(f" Dataset validation passed:")
        print(f"  - Train: {dataset_info.get('train')}")
        print(f"  - Val  : {dataset_info.get('val')}")
        print(f"  - Test : {dataset_info.get('test')}")
        print(f"  - Classes ({dataset_info.get('nc')}): {dataset_info.get('names')}")
    except Exception as e:
        print(f"ERROR: Dataset verification failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Hardware / Device Selection
    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 2] Selected hardware device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 3. Model Loading
    print(f"\n[Step 3] Loading YOLO26 base model: {model_path}...")
    try:
        model = YOLO(str(model_path))
        print(f" YOLO26 model loaded successfully: {model.task} model")
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve project output path to ensure isolated run folder
    abs_project_dir = str(project_root / args.project if not os.path.isabs(args.project) else Path(args.project))
    print(f" Output project directory: {abs_project_dir}")
    print(f" Run experiment name     : {args.name}")

    # Safety check on root best.pt
    root_best = project_root / "best.pt"
    if root_best.exists():
        root_best_mtime = root_best.stat().st_mtime
        print(f" Protected root checkpoint found: {root_best} (Preserved)")
    else:
        root_best_mtime = None

    # 4. Training
    print("\n[Step 4] Starting YOLO26 training...")
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            project=abs_project_dir,
            name=args.name,
            exist_ok=args.exist_ok,
            fraction=args.fraction
        )
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Post-training Verification
    new_weights_dir = Path(abs_project_dir) / args.name / "weights"
    new_best = new_weights_dir / "best.pt"
    new_last = new_weights_dir / "last.pt"

    print("\n=================================================================")
    print("                     TRAINING COMPLETE                           ")
    print("=================================================================")
    if new_best.exists():
        print(f" New best checkpoint saved: {new_best} ({new_best.stat().st_size / (1024*1024):.2f} MB)")
    if new_last.exists():
        print(f" New last checkpoint saved: {new_last} ({new_last.stat().st_size / (1024*1024):.2f} MB)")

    # Verify root best.pt was never altered
    if root_best.exists():
        current_mtime = root_best.stat().st_mtime
        if current_mtime == root_best_mtime:
            print(" Existing root 'best.pt' remained completely UNCHANGED.")
        else:
            print(" WARNING: Root 'best.pt' timestamp modified!", file=sys.stderr)

    return results

if __name__ == "__main__":
    main()
