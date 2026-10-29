import argparse
import os
from ultralytics import YOLO

def main(model_path, source, save_result, conf_thres):
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' was not found.")
        print("If you haven't trained YOLO26 yet, run train_yolo26.py first or use --model best.pt")
        return

    print(f"Loading model: {model_path}")
    # Load the trained model
    model = YOLO(model_path)
    
    print(f"Running inference on source: {source}")
    # Run prediction
    results = model.predict(source=source, save=save_result, conf=conf_thres)
    
    # Process results
    for result in results:
        # Show the image with bounded boxes
        result.show()
        
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"\nDetected {len(boxes)} objects:")
            # Print class names and confidences
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"- {class_name} (Confidence: {conf:.2f})")
        else:
            print("\nNo objects detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO object detection inference.")
    
    # Determine the default model based on existence of trained yolo26
    default_model = os.path.join("runs", "detect", "yolo26_garbage", "weights", "best.pt")
    if not os.path.exists(default_model):
        default_model = "best.pt"
        
    parser.add_argument("--model", type=str, default=default_model, help="Path to the model weights (e.g., best.pt)")
    parser.add_argument("--source", type=str, default=None, help="Source for inference: path to image/video or '0' for webcam")
    parser.add_argument("--save", action="store_true", help="Save the output image/video with bounding boxes")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction")
    
    args = parser.parse_args()
    
    # Prompt for input in the terminal if no source was provided via arguments
    if args.source is None:
        while True:
            source_input = input("Enter the path of the image/video file (or '0' for webcam): ").strip()
            
            # Remove surrounding quotes if the user pasted a path with them
            if source_input.startswith('"') and source_input.endswith('"'):
                source_input = source_input[1:-1]
            elif source_input.startswith("'") and source_input.endswith("'"):
                source_input = source_input[1:-1]
                
            if source_input == '0' or os.path.exists(source_input):
                args.source = source_input
                break
            else:
                print(f"Error: The file '{source_input}' does not exist. Please try again.")
    
    main(args.model, args.source, args.save, args.conf)
