from ultralytics import YOLO

if __name__ == '__main__':
    # Load your newly trained weights
    # (Check your exact 'runs' folder path to ensure this matches)
    model = YOLO("runs/detect/yolo26_local_run/weights/best.pt")
    
    print("Exporting NMS-Free model...")
    # Export to ONNX format (NMS=False is natively handled by YOLO26)
    export_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True
    )
    print(f"Export complete! File saved at: {export_path}")
