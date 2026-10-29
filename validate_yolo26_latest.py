"""
validate_yolo26_latest.py
Validation and evaluation script for YOLO26 Garbage Detection model.
Evaluates precision, recall, mAP50, mAP50-95, inference speed, and confusion matrix
on both the validation set and the held-out test set.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def evaluate_split(model, data_yaml, split_name, project_dir, run_name, device):
    print(f"\n" + "="*60)
    print(f" EVALUATING ON {split_name.upper()} SET")
    print("="*60)

    val_args = {
        "data": str(data_yaml),
        "split": split_name,
        "device": device if device else None,
        "project": str(project_dir),
        "name": f"{run_name}_{split_name}",
        "exist_ok": True,
        "plots": True
    }

    metrics = model.val(**val_args)

    precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
    recall = metrics.results_dict.get('metrics/recall(B)', 0.0)
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)

    speed_info = metrics.speed if hasattr(metrics, 'speed') else {}
    preprocess_speed = speed_info.get('preprocess', 0.0)
    inference_speed = speed_info.get('inference', 0.0)
    loss_speed = speed_info.get('loss', 0.0)
    postprocess_speed = speed_info.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed

    print(f"\n--- {split_name.upper()} RESULTS ---")
    print(f"Precision (B)  : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (B)     : {recall:.4f} ({recall*100:.2f}%)")
    print(f"mAP@50 (B)     : {map50:.4f} ({map50*100:.2f}%)")
    print(f"mAP@50-95 (B)  : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"Speed Details  : Preprocess={preprocess_speed:.2f}ms, Inference={inference_speed:.2f}ms, Postprocess={postprocess_speed:.2f}ms")
    print(f"Total Speed    : {total_speed:.2f}ms per image ({1000.0/total_speed:.1f} FPS)" if total_speed > 0 else "")

    # Per-class metrics
    class_names = model.names
    print(f"\nPer-class results:")
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for cls_idx, cls_map in enumerate(metrics.box.maps):
            cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
            print(f" - {cls_name} (Class {cls_idx}): mAP50-95 = {cls_map:.4f}")

    # Confusion matrix location
    save_dir = Path(metrics.save_dir) if hasattr(metrics, 'save_dir') else None
    cm_path = save_dir / "confusion_matrix.png" if save_dir else None
    if cm_path and cm_path.exists():
        print(f" Confusion matrix plot: {cm_path}")

    return {
        "split": split_name,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "speed": speed_info,
        "save_dir": str(save_dir) if save_dir else "",
        "confusion_matrix": str(cm_path) if cm_path and cm_path.exists() else ""
    }

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO26 Garbage Detection Model")
    parser.add_argument("--model", type=str,
                        default="runs/garbage_detection/yolo26n_baseline/weights/best.pt",
                        help="Path to trained YOLO26 checkpoint")
    parser.add_argument("--data", type=str, default="datasets/garbage_yolo26/data.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--split", type=str, default="both",
                        choices=["val", "test", "both"],
                        help="Split to evaluate: 'val', 'test', or 'both' (default: both)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '0' for GPU, 'cpu', or leave empty")
    parser.add_argument("--project", type=str, default="runs/garbage_detection",
                        help="Project directory for validation runs")
    parser.add_argument("--name", type=str, default="yolo26n_eval",
                        help="Run subfolder name")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    model_path = project_root / args.model if not os.path.isabs(args.model) else Path(args.model)
    data_yaml = project_root / args.data if not os.path.isabs(args.data) else Path(args.data)
    project_dir = project_root / args.project if not os.path.isabs(args.project) else Path(args.project)

    print("=================================================================")
    print("           YOLO26 MODEL VALIDATION & EVALUATION                  ")
    print("=================================================================")
    print(f"Model File   : {model_path}")
    print(f"Dataset YAML : {data_yaml}")
    print(f"Device       : {args.device if args.device else 'auto'}")
    print(f"Target Split : {args.split}")

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at: {model_path}", file=sys.stderr)
        print("Please train the model first using train_yolo26_latest.py", file=sys.stderr)
        sys.exit(1)

    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found at: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model: {model_path}...")
    model = YOLO(str(model_path))

    results = {}
    if args.split in ["val", "both"]:
        results["val"] = evaluate_split(model, data_yaml, "val", project_dir, args.name, args.device)

    if args.split in ["test", "both"]:
        results["test"] = evaluate_split(model, data_yaml, "test", project_dir, args.name, args.device)

    print("\n=================================================================")
    print("                    VALIDATION SUMMARY                           ")
    print("=================================================================")
    for split_k, r in results.items():
        print(f"[{split_k.upper()}] P: {r['precision']:.4f} | R: {r['recall']:.4f} | mAP50: {r['map50']:.4f} | mAP50-95: {r['map50_95']:.4f}")

    return results

if __name__ == "__main__":
    main()
