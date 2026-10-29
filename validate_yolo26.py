import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO26 for Garbage Detection")
    parser.add_argument("--model", type=str, default=os.path.join("runs", "detect", "yolo26_garbage", "weights", "best.pt"), 
                        help="Path to the trained model (e.g., runs/detect/yolo26_garbage/weights/best.pt or best.pt)")
    parser.add_argument("--device", type=str, default="", help="Device to use for validation (e.g., '0' for GPU, 'cpu')")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_dir, "datasets", "garbage_yolo26", "data.yaml")
    images_val_dir = os.path.join(project_dir, "datasets", "garbage_yolo26", "valid", "images")
    labels_val_dir = os.path.join(project_dir, "datasets", "garbage_yolo26", "valid", "labels")

    if not os.path.exists(args.model):
        print(f"Error: The model file '{args.model}' was not found.")
        print("Make sure you have trained the model first, or specify a valid --model path (like best.pt).")
        return

    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset configuration '{data_yaml_path}' is missing!")
        return

    if not os.path.exists(images_val_dir) or not os.path.exists(labels_val_dir):
        print("Error: The validation dataset images or labels are missing.")
        print("Please ensure they are in 'datasets/garbage_yolo26/valid/'.")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Starting validation...")
    metrics = model.val(
        data=data_yaml_path,
        device=args.device if args.device else None,
        project="runs/detect",
        name="yolo_val",
        exist_ok=True
    )
    
    print("\n--- Validation Results ---")
    if hasattr(metrics, 'results_dict'):
        print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
        print(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
        print(f"mAP@50: {metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
        print(f"mAP@50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    
    if hasattr(metrics, 'speed'):
        inference_speed = metrics.speed.get('inference', 0.0)
        print(f"Inference Speed: {inference_speed:.2f} ms per image")

if __name__ == "__main__":
    main()
