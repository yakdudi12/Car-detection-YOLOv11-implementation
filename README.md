# Car-detection-YOLOv11-implementation
Vehicle detection project in videos using YOLOv11. It allows batch capture of traffic or parking videos, identifying the presence of cars using a fine-tuned model and automatically saving the results for later analysis. It allows training a model, performing inference on single videos, and processing multiple videos in batch automatically.
```
Project Structure
📂 YOLO Car Inference  
 ├── 📄 YOLO_CD_train.py          # Main py training 
 ├── 📄 YOLO_CD_inference.py      # Inference with pre-trained model (1 video)
 ├── 📄 YOLO_CD_script.py         # Py code for larger-scale production (>1 videos)
 ├── 📄 run_all_videos.bat        # Script customizable to MLops
 ├── 📄 README.md                 # This file
 ├── 📂 Plots & analitics         # All training and performance data
 ├── 📂 trafic_data
      ├──📂 train                 # Train images and Labels in YOLO format
      ├──📂 valid                 # Validation images and Labels in YOLO format
      └──📄 data_1.yaml           # Data yaml ready to use 
 ├── 📂 outputs                   # Video outputs (after inference)
 ├── 📂 videos                    # Video inputs 
 └── 📂 models                    # trained model
      └──📄 best.pt               # Result from YOLO_CD_train.py
 ```
🧰 Libraries Used:
```python
import os
from ultralytics import YOLO (yolov11)
import torch
import sys
```
How to use:
🏋️‍♂️ Model Training
To train YOLOv11 on your dataset defined in data_1.yaml:
```bash
      python YOLO_CD_train.py
```
```
Key parameters:
  Base model: yolo11n.pt
  Epochs: 50
  Image size: 640x640
  Batch size: 16
  Device: GPU if available, otherwise CPU
```
🎥 Single Video Inference
The script yolo_inference.py allows using a pretrained model (best.pt) to process a single video:
```python
trained_model = r"models/best.pt"
video_path = r"videos/Parking Garage Rooftop Security.mp4"

if __name__ == "__main__":
  results = YOLO(trained_model)(
      video_path,
      save=True,
      project="outputs",
      name="cardetection_runtest")
```
📦 Batch Video Inference
The script YOLO_CD_script.py can process multiple videos automatically:
```bash
python yolo_inference_batch.py <path_model> <path_video> <output_folder>
```
```
Parameters:
<path_model>: path to the trained model (best.pt)
<path_video>: path to the video or folder containing videos
<output_folder>: folder where results will be saved
```
A run_inference.bat file is included to run the batch inference for all .mp4 videos in a folder. Ready to implement in MLops Pipelines.
Automating with .bat:
```bash
@echo off
REM venv
call "your favorite venv"

REM Videos Folder
set VIDEO_FOLDER= Path to the video inputs Folder

REM Output Folder
set OUTPUT_FOLDER= Path to the video outputs Folder

REM Trained Model
set MODEL_PATH= Path to the model trained cache "best.pt"

REM Process all videos .mp4
for %%f in ("%VIDEO_FOLDER%\*.mp4") do (
    echo Processing %%f ...
    py "Path to YOLO_CD_script.py" "%MODEL_PATH%" "%%f" "%OUTPUT_FOLDER%"
)

echo All videos processed ;)
pause
```
📈 Results
Annotated videos with bounding boxes around detected vehicles.
Training metrics: precision, recall, mAP.
Example output:
![Demo](outputs/Results1.gif)
![Demo](outputs/results2.gif)
<img width="1912" height="1079" alt="image" src="https://github.com/user-attachments/assets/92d03663-26c2-4f71-9884-09ba0f17b214" />
![val_batch0_pred](https://github.com/user-attachments/assets/cb522236-76a6-4588-bee8-53548047e938)
![val_batch1_labels](https://github.com/user-attachments/assets/d1c81060-cdf3-46fd-aac3-c7cd5ae86ec7)
<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/bea61dcc-5d16-4301-baf4-72e5f26f5533" />


✒️ References
* Train dataset: Road Vehicle Images Dataset https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset/data (Ashfak Yeafi)
* Video 1: https://www.youtube.com/watch?v=zOq2XdwHGT0&ab_channel=FreeStockVideos
* Video 2: https://www.youtube.com/watch?v=wWLAc6mdJrs&t=2s&ab_channel=mjrzeman
* Video 3: https://www.youtube.com/watch?v=u4SzzYX5HoI&ab_channel=KilmerMedia

🤝 Contributions

This project is part of my professional portfolio. If you have suggestions or feedback, I would be happy to hear from you!

📬 Contact

Britez Santiago
[LinkedIn](https://www.linkedin.com/in/santiago-luis-britez-101a8a217)
