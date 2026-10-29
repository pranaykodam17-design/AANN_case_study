"""
detect_select_image.py
YOLO26 Garbage Detection Script.

Workflow:
1. Opens a native Windows file selection dialog (tkinter.filedialog.askopenfilename).
2. Loads trained YOLO26 model from runs/garbage_detection/yolo26n_baseline_100ep/weights/best.pt.
3. Runs CPU inference (imgsz=640, conf=0.25).
4. Saves the annotated output image uniquely in runs/predictions/interactive_test/.
5. Prints detection count, confidences, and output path in the terminal.
6. Automatically opens the saved image in the default Windows image viewer via os.startfile.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Trained checkpoint and output directory configuration
MODEL_RELATIVE_PATH = "runs/garbage_detection/yolo26n_baseline_100ep/weights/best.pt"
OUTPUT_DIR_NAME = "runs/predictions/interactive_test"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_model_checkpoint() -> Path:
    """
    Locates the trained YOLO26 checkpoint.
    Ensures that the actual trained checkpoint is used, not root-level best.pt or yolo26n.pt.
    Exits with an informative error if the checkpoint is missing.
    """
    script_dir = Path(__file__).resolve().parent

    candidate_paths = [
        Path(MODEL_RELATIVE_PATH),
        script_dir / MODEL_RELATIVE_PATH,
        script_dir / "Garbage-detection" / MODEL_RELATIVE_PATH,
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate.resolve()

    expected_path = (script_dir / MODEL_RELATIVE_PATH).resolve()
    print(f"\nError: Trained model checkpoint not found at:\n  {expected_path}", file=sys.stderr)
    print(f"Please verify that '{MODEL_RELATIVE_PATH}' exists.", file=sys.stderr)
    sys.exit(1)


def open_file_picker() -> str:
    """
    Opens a native Windows file selection dialog using tkinter.filedialog.askopenfilename.
    Supports .jpg, .jpeg, .png, .bmp, and .webp image formats.
    Returns the selected file path, or an empty string if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.focus_force()

    file_types = [
        ("Supported Image Files (*.jpg;*.jpeg;*.png;*.bmp;*.webp)", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("JPEG Images (*.jpg; *.jpeg)", "*.jpg *.jpeg"),
        ("PNG Images (*.png)", "*.png"),
        ("BMP Images (*.bmp)", "*.bmp"),
        ("WEBP Images (*.webp)", "*.webp"),
        ("All Files (*.*)", "*.*"),
    ]

    selected_file = filedialog.askopenfilename(
        title="Select an Image for Garbage Detection",
        filetypes=file_types
    )

    root.destroy()
    return selected_file


def save_annotated_image(annotated_img: np.ndarray, source_image_path: Path, output_base_dir: Path) -> Path:
    """
    Saves the annotated image to runs/predictions/interactive_test/.
    Uses the original image filename as the base.
    Generates a unique filename if an output file with the same name already exists.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)

    stem = source_image_path.stem
    ext = source_image_path.suffix
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        ext = ".jpg"

    candidate = output_base_dir / f"{stem}{ext}"
    counter = 1
    while candidate.exists():
        candidate = output_base_dir / f"{stem}_{counter}{ext}"
        counter += 1

    resolved_output = candidate.resolve()
    success = cv2.imwrite(str(resolved_output), annotated_img)
    if not success:
        print(f"\nError: Failed to write output image to: {resolved_output}", file=sys.stderr)
        sys.exit(1)

    return resolved_output


def create_annotated_display(result, model) -> tuple[np.ndarray, int, float]:
    """
    Creates the display image showing:
    - Original image with bounding boxes
    - Garbage class labels
    - Confidence percentages
    - Number of detected objects
    - Inference time
    - "No garbage detected" banner if no objects found
    """
    num_detections = len(result.boxes)
    inf_time_ms = result.speed.get("inference", 0.0)

    # Base plot from ultralytics: bounding boxes, class labels, confidences
    annotated = result.plot(conf=True, labels=True, line_width=2)

    h, w = annotated.shape[:2]

    # Create top status HUD banner
    banner_height = max(50, min(80, int(h * 0.10)))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 30), -1)

    # Accent line at bottom of banner
    border_color = (0, 220, 110) if num_detections > 0 else (0, 140, 255)
    cv2.line(overlay, (0, banner_height), (w, banner_height), border_color, 2)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.9, w / 900.0))
    thickness = 2
    y_pos = int(banner_height * 0.65)

    if num_detections > 0:
        status_text = f"Garbage Detected: {num_detections} object(s)"
        status_color = (0, 255, 128)
    else:
        status_text = "No garbage detected"
        status_color = (0, 165, 255)

    time_text = f"Inference Time: {inf_time_ms:.1f} ms"

    cv2.putText(annotated, status_text, (16, y_pos), font, font_scale, status_color, thickness, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
    x_pos = max(w - tw - 16, 20 + int(w * 0.45))
    cv2.putText(annotated, time_text, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # If no garbage detected, also draw prominent centered notification
    if num_detections == 0:
        center_text = "No garbage detected"
        c_font_scale = max(0.8, min(1.3, w / 600.0))
        (ctw, cth), _ = cv2.getTextSize(center_text, font, c_font_scale, 2)
        cx = (w - ctw) // 2
        cy = (h + cth) // 2
        pad = 20
        box_overlay = annotated.copy()
        cv2.rectangle(box_overlay, (cx - pad, cy - cth - pad), (cx + ctw + pad, cy + pad), (20, 20, 20), -1)
        cv2.rectangle(box_overlay, (cx - pad, cy - cth - pad), (cx + ctw + pad, cy + pad), (0, 140, 255), 2)
        cv2.addWeighted(box_overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.putText(annotated, center_text, (cx, cy), font, c_font_scale, (0, 215, 255), 2, cv2.LINE_AA)

    return annotated, num_detections, inf_time_ms


def launch_default_viewer(output_path: Path):
    """
    Immediately launches the saved output image using os.startfile on Windows
    so the user can immediately see the result in the default image viewer (Photos).
    If it fails, displays a clear error message containing the output path.
    """
    print(f"\n[4/4] Opening annotated image in default Windows viewer...")
    try:
        os.startfile(str(output_path))
        print(f"      Launched: {output_path}")
    except Exception as e:
        print(f"\nError: Could not launch the Windows image viewer for:\n  {output_path}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Garbage Detection with Windows File Picker")
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Optional path to image file (if omitted, opens native Windows file picker dialog)"
    )
    parser.add_argument(
        "--show-cv2", action="store_true", default=False,
        help="Optional: Also display the annotated image in an OpenCV window"
    )
    args = parser.parse_args()

    # Step 1: Check and locate trained model checkpoint
    model_path = find_model_checkpoint()

    # Step 2: Open Windows file picker selection dialog
    if args.image:
        selected_file = args.image.strip("\"' \t\r\n`")
    else:
        print("Opening file selection dialog...")
        selected_file = open_file_picker()

    # Allow cancelling safely without crashing
    if not selected_file:
        print("No image selected. Exiting.")
        sys.exit(0)

    selected_image_path = Path(selected_file).resolve()

    # Error handling: File does not exist
    if not selected_image_path.exists() or not selected_image_path.is_file():
        print(f"\nError: The specified file does not exist:\n  {selected_image_path}", file=sys.stderr)
        sys.exit(1)

    # Error handling: Unsupported file format
    if selected_image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(ext.upper().lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        print(f"\nError: Unsupported file format '{selected_image_path.suffix}'.", file=sys.stderr)
        print(f"Supported image formats are: {supported_str}.", file=sys.stderr)
        sys.exit(1)

    # Error handling: Unreadable or corrupted image
    verify_img = cv2.imread(str(selected_image_path))
    if verify_img is None:
        print(f"\nError: Unable to open or decode image: '{selected_image_path}'.", file=sys.stderr)
        print("The file may be corrupted or not a valid image file.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Load trained model checkpoint (using CPU inference)
    print(f"\n[1/4] Loading trained YOLO26 model from:")
    print(f"      {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"\nError: Failed to load model checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 4: Run inference
    print(f"\n[2/4] Running YOLO26 prediction on:")
    print(f"      {selected_image_path}")
    print(f"      Settings: imgsz=640, conf=0.25, device='cpu'")

    try:
        results = model.predict(
            source=str(selected_image_path),
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )
        result = results[0]
    except Exception as e:
        print(f"\nError during YOLO26 inference: {e}", file=sys.stderr)
        sys.exit(1)

    # Create annotated image with bounding boxes, labels, confidences, and status HUD
    annotated_img, num_detections, inf_time_ms = create_annotated_display(result, model)

    # Step 5: Save output in runs/predictions/interactive_test/
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_DIR_NAME
    saved_path = save_annotated_image(annotated_img, selected_image_path, output_dir)
    print(f"\n[3/4] Saved annotated image:")
    print(f"      {saved_path}")

    # Step 6: Print detection results and object details in the terminal
    print("\n" + "=" * 50)
    print("              DETECTION RESULTS")
    print("=" * 50)
    print(f"Input Image          : {selected_image_path}")
    print(f"Total Detections     : {num_detections}")
    print(f"Inference Time       : {inf_time_ms:.1f} ms")
    print(f"Saved Output Path    : {saved_path}")
    print("-" * 50)

    if num_detections > 0:
        print("Detected Objects Breakdown:")
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            print(f"  [{idx + 1}] Class: '{cls_name}' | Confidence: {conf * 100:.2f}% | Box (xyxy): {xyxy}")
    else:
        print("No garbage detected")
    print("=" * 50)

    # Step 7: Immediately launch the saved output image using os.startfile on Windows
    launch_default_viewer(saved_path)

    # Optional OpenCV window if explicitly requested
    if args.show_cv2:
        window_title = f"YOLO26 - {num_detections} detected" if num_detections > 0 else "YOLO26 - No garbage"
        cv2.imshow(window_title, annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
