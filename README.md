# 🚨 Collision, Fire, and Smoke Detection Model using YOLOv11
This repository contains object detection models trained to detect **accidents** and **natural disasters** using YOLOv11. The system is part of a broader initiative to improve real-time disaster and accident response through automated monitoring of CCTV feeds. The model training includes detection for car collision, fire, and smoke.

### 🧠 Model Descriptions
- **custom.pt**: A custom YOLOv11-based model trained on 25,000 labeled images across 3 classes (collision, fire, smoke). Designed specifically for accident and disaster detection using CCTV footage.
- **yolov11n.pt**: A pre-trained YOLOv11 nano model trained on the COCO dataset. Lightweight and optimized for general-purpose object detection with high inference speed, but not tailored for disaster-specific scenarios.

## Data Balance Summary
---------------------------------------------------------
Collision
📊 Dataset Split Summary
Total data: 6251

Fire
📊 Dataset Split Summary
Total data: 6252

🚗 Collision: 12500
🔥 Fire: 12500
🧮 Overall Total: 25000 images

| Metric         | Value               |
|----------------|---------------------|
| Precision (B)  | 0.6225728016728798  |
| Recall (B)     | 0.592293396436181   |
| mAP50 (B)      | 0.587249281543145   |
| mAP50-95 (B)   | 0.4040384620980947  |
| Fitness        | 0.4217960240991292  |

## 🎥 Detection Demo
<p align="center">
  <img src="1-collision-demo.gif" width="100%"/><br><br>
  <img src="2-fire-demo.gif" width="100%"/>
</p>
