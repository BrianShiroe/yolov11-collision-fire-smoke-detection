#train code
from ultralytics import YOLO

# model path / use last.pt model when resuming from previous training
model_path = "model/yolo11n.pt"

# Load the pre-trained YOLO model
model = YOLO(model_path)

if __name__ == "__main__":
    # Train the model with the new data (e.g., car crashes) while retaining original classes
    model.train(
        data="data.yaml",  # Custom dataset config
        imgsz=640,  # Image size
        batch=16,  # Batch size
        epochs=100,  # Training epochs
        workers=8,  # Data loading workers
        device='0',  # GPU (0) or CPU (1)
        amp=False, # Disable Automatic Mixed Precision to prevent NaNs on GTX 1650
        # resume=False  # Resume from last checkpoint
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model