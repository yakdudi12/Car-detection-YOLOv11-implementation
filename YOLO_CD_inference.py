import os
from ultralytics import YOLO
import torch

trained_model = r""
video_path = r""
if __name__ == "__main__":
    # Inference Video
    results = YOLO(trained_model)(video_path,
                    save=True,
                    project=r"\YOLO Car detection",
                    name="cardetection_runtest3")

    print("End of the inference")