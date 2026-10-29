"""
test_yolo26_latest.py
Runs inference on real images using the newly trained YOLO26 checkpoint.
Tests:
- Standard test images
- Difficult images with multiple objects/clutter
- Clean images with no garbage (negative controls)
- Root workspace test image (test.jpg)
Saves annotated visual outputs to outputs/predictions/ and logs confidence scores.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_image_inference(model, image_path, output_dir, conf_threshold=0.25):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Warning: Image {image_path} does not exist. Skipping.")
        return None

    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )

    res = results[0]
    boxes = res.boxes
    detections = []

    print(f"\n--- Testing: {img_path.name} ---")
    print(f"Path: {img_path}")
    print(f"Detected Objects: {len(boxes)}")

    if len(boxes) == 0:
        print("  -> No objects detected (clean / below threshold).")
    else:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            detections.append({
                "index": idx + 1,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": xyxy
            })
            print(f"  [{idx+1}] Class: '{cls_name}' | Confidence: {conf:.4f} | Box (xyxy): {xyxy}")

    return {
        "image": str(img_path),
        "filename": img_path.name,
        "count": len(boxes),
        "detections": detections,
        "speed": res.speed
    }

def main():
    parser = argparse.ArgumentParser(description="Test YOLO26 on Real Images")
    parser.add_argument("--model", type=str,
                        default="runs/garbage_detection/yolo26n_baseline/weights/best.pt",
                        help="Path to trained YOLO26 checkpoint")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Confidence threshold for predictions (default: 0.20)")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions",
                        help="Directory to save annotated test outputs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = project_root / args.model if not os.path.isabs(args.model) else Path(args.model)
    output_dir = project_root / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)

    print("=================================================================")
    print("             YOLO26 REAL-IMAGE INFERENCE TESTING                 ")
    print("=================================================================")
    print(f"Model File       : {model_path}")
    print(f"Output Directory : {output_dir}")
    print(f"Confidence Thresh: {args.conf}")

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at: {model_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    # Selected representative test images
    test_cases = [
        # 1. Standard test images
        {
            "category": "Standard Test Image 1",
            "path": project_root / "datasets/garbage_yolo26/test/images/000144244W_jpg.rf.d1dc6442492ffbc7bcce5f0412bfea66.jpg"
        },
        {
            "category": "Standard Test Image 2",
            "path": project_root / "datasets/garbage_yolo26/test/images/02f330f2-9439-11e5-982d-40f2e96c8ad8_jpg.rf.af9f304e5a78fc6f84d268edef376fe3.jpg"
        },
        # 2. Cluttered / Difficult images with multiple objects
        {
            "category": "Difficult Cluttered Scene (Multiple items)",
            "path": project_root / "datasets/garbage_yolo26/test/images/3adba2c8-6799-11e5-8dc4-40f2e96c8ad8_jpg.rf.47a0aeb81277b3f2424793ac07bf702b.jpg"
        },
        {
            "category": "Difficult Cluttered Scene 2",
            "path": project_root / "datasets/garbage_yolo26/test/images/a210c932-679e-11e5-b0d3-40f2e96c8ad8_jpg.rf.67b3ba526a4be2988327eaa291455e56.jpg"
        },
        # 3. Negative control image (Clean background - No garbage)
        {
            "category": "Negative Control (Zero Garbage / Clean Background)",
            "path": project_root / "datasets/garbage_yolo26/test/images/0ed6fe62-9439-11e5-982d-40f2e96c8ad8_jpg.rf.632814891b404ce212650bdc302e8852.jpg"
        },
        # 4. Root test image
        {
            "category": "Workspace Root test.jpg",
            "path": project_root / "test.jpg"
        }
    ]

    all_results = []
    for item in test_cases:
        print(f"\nEvaluating Category: {item['category']}")
        res = run_image_inference(model, item["path"], output_dir, conf_threshold=args.conf)
        if res:
            res["category"] = item["category"]
            all_results.append(res)

    print("\n=================================================================")
    print("                     INFERENCE COMPLETED                         ")
    print(f"Results saved to: {output_dir}")
    print("=================================================================")

    return all_results

if __name__ == "__main__":
    main()
