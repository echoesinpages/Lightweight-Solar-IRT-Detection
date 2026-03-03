from ultralytics import YOLO

if __name__ == '__main__':
    # Load the base YOLO26 architecture
    model = YOLO("yolo26n.pt")

    # Start training the AI
    results = model.train(
        data="dataset_yolo/pvel_ad.yaml", # This points to the folder you just created
        epochs=300, 
        imgsz=640, 
        batch=16,          
        optimizer="MuSGD", 
        lr0=0.01,
        momentum=0.9,
        workers=0,         # This MUST be 0 on Windows to prevent freezing
        project="Solar_Defect_Detection",
        name="yolo26_local_run"
    )