# 🌾 Rice Disease Detection using YOLOv8

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Problem Statement
Rice is one of India's most critical crops. Diseases like Bacterial Leaf 
Blight, Brown Spot, and Leaf Blast can devastate yields if not caught early.
This project builds a real-time detection system using YOLOv8 to localize 
and classify disease regions on rice leaves — enabling farmers to get 
instant, precise diagnosis from a phone camera.

## Demo
> GIF coming after training

## Classes Detected
| Class | Description |
|-------|-------------|
| Bacterial Leaf Blight | Water-soaked lesions along leaf margins |
| Brown Spot | Circular brown spots, often with yellow halo |
| Leaf Blast | Diamond-shaped lesions, gray-white center |
| Healthy | No disease present |

## Architecture
- **Model**: YOLOv8n (nano) for speed, upgradeable to YOLOv8s/m
- **Dataset**: [Roboflow — link coming]
- **Training**: Google Colab (T4 GPU)
- **Inference**: GitHub Codespace / CPU

## Results
| Model | mAP@50 | Precision | Recall |
|-------|--------|-----------|--------|
| YOLOv8n | TBD | TBD | TBD |

## Project Structure
yolov8-custom-detection/
├── data/               # Dataset (downloaded via Roboflow, not committed)
├── notebooks/          # Colab training notebook
├── src/                # Inference scripts
├── assets/             # Demo GIFs and architecture diagrams
└── data.yaml           # Dataset config for YOLOv8

## Setup & Run
```bash
pip install -r requirements.txt
python src/predict.py --source your_image.jpg
```


## What I Learned
- How YOLOv8's anchor-free head differs from earlier YOLO versions
- Why including a "Healthy" class prevents false positive detections
- How annotation quality affects mAP more than model size
- Tradeoffs between YOLOv8n vs YOLOv8s for edge deployment

## Dataset
Downloaded from Roboflow. Annotated with bounding boxes around 
disease regions. ~N images across 4 classes.

start