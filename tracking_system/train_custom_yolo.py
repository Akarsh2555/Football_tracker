from roboflow import Roboflow
from ultralytics import YOLO
import os

def download_and_train():
    print("=== Downloading Dataset from Roboflow ===")
    
    # 1. Download Dataset
    # -------------------------------------------------------------------------------------
    # USER TODO: Replace the block below with the Python snippet from Roboflow Universe
    # (Click "Download Dataset" -> Format: "YOLOv8" -> "Show Download Code")
    rf = Roboflow(api_key="XmL0vsfrtZpOUNzMaizm")
    project = rf.workspace("augmented-startups").project("football-player-detection-kucab")
    version = project.version(8)
    dataset = version.download("yolov8")
    
    print("\nDataset Downloaded to:", dataset.location)
    
    # 2. Train Custom YOLOv8 Model
    # -------------------------------------------------------------------------------------
    print("\n=== Starting YOLOv8 Custom Training ===")
    
    # Load the base pre-trained model
    model = YOLO("yolov8n.pt") 
    
    # Define the path to the data.yaml file inside the downloaded Roboflow dataset
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    print(f"Training on dataset config: {data_yaml_path}")
    
    # Start training! 
    # (epochs=20 is a quick test, but you usually want 50-100 for good results)
    results = model.train(
        data=data_yaml_path,
        epochs=30,           # Number of times it loops over the dataset
        imgsz=640,           # Image size
        batch=16,            # Batch size (lower this if your computer runs out of memory)
        project="custom_yolo", # Folder to save results
        name="football_tracker", # Name of the model run
        freeze=10            # Freeze backbone (first 10 layers), only train the head
    )
    
    print("\n=== Training Complete! ===")
    print("Your new custom model weights are saved in: custom_yolo/football_tracker/weights/best.pt")
    
if __name__ == "__main__":
    download_and_train()
