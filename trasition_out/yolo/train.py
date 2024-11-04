import argparse
from ultralytics import YOLO

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Train a YOLO model using Ultralytics YOLOv8')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    args = parser.parse_args()

    # Load a model
    model = YOLO("yolov8n.pt")  # assuming yolov8n.pt is the correct model file

    # Train the model
    results = model.train(data="coco.yaml", epochs=args.epochs, batch=args.batch_size, imgsz=640)

    # Optionally, print out results or any other information
    print("Training completed with the following results:")
    print(results)

if __name__ == "__main__":
    main()
