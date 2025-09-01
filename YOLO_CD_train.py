import os
from ultralytics import YOLO
import torch

dataset_root = r"" #Path trafic_data
yaml_path = os.path.join(dataset_root, "data_1.yaml")   #YAML


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CUDA implementation

    #Model selection
    model = YOLO("yolo11n.pt").to(device) #Model Yolo11

    #Model Training
    model.train(data=yaml_path,epochs=50,imgsz=640,batch=16,workers=4)

    # Inference Video
    video_path = r""
    results = model(video_path,
                    save=True,
                    project=r"\YOLO Car detection",
                    name="cardetection_runtest1")

    print("End of the training and inference")