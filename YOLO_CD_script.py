import os
import sys
from ultralytics import YOLO

def main(model_path, video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model = YOLO(model_path)

    results = model(
        video_path,
        save=True,
        project=output_folder,
        name=video_name
    )


    print(f"Processed video: {video_name}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python yolo_inference.py <path_model> <path_video> <output>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]
    output_folder = sys.argv[3]

    main(model_path, video_path, output_folder)