import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 for Garbage Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device to use for training (e.g., '0' for GPU, 'cpu')")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base model to start training from")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_dir, "datasets", "garbage_yolo26", "data.yaml")
    images_train_dir = os.path.join(project_dir, "datasets", "garbage_yolo26", "train", "images")
    labels_train_dir = os.path.join(project_dir, "datasets", "garbage_yolo26", "train", "labels")

    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} is missing!")
        return

    if not os.path.exists(images_train_dir) or not os.path.exists(labels_train_dir):
        print("===============================================================")
        print("ERROR: Dataset images or labels are missing!")
        print(f"Could not find the training directory: {images_train_dir}")
        print("Please ensure your dataset is properly extracted and structured:")
        print("  datasets/garbage_yolo26/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("===============================================================")
        return

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print("Starting training...")
    model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device if args.device else None,
        project="runs/detect",
        name="yolo26_garbage",
        exist_ok=True
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {os.path.join('runs', 'detect', 'yolo26_garbage', 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
    