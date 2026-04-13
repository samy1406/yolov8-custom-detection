import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    
    # --source: Path to the image or directory
    parser.add_argument("--source", type=str, required=True, 
                        help="Path to input image or folder")
    
    # --weights: Path to your trained model (defaulting to best.pt)
    parser.add_argument("--weights", type=str, default="best.pt", 
                        help="Path to weights file (default: best.pt)")
    
    # --conf: Confidence threshold (defaulting to our discussed 0.40)
    parser.add_argument("--conf", type=float, default=0.40, 
                        help="Confidence threshold (default: 0.40)")
    
    # --save: A 'flag' that is True if --save is present, False otherwise
    parser.add_argument("--save", action="store_true", 
                        help="Save the predicted images to disk")

    return parser.parse_args()


# Example of how you'd call it in the script
if __name__ == "__main__":
    args = parse_args()
    print(f"Loading model: {args.weights} with confidence: {args.conf}")
    model = YOLO(args.weights)
    print(f"Model loaded successfully from {args.weights}")
    # Run the prediction
    results = model.predict(
        source=args.source, 
        conf=args.conf, 
        save=args.save
    )
    # Iterate through the list of results (one per image)
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            print(f"No disease detected in: {result.path}")
            continue
            
        for box in boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"Image: {result.path} | Detected: {label} | Conf: {confidence:.2f} | Box: {coords}")


