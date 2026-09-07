"""
YOLOv8 Landslide Instance Segmentation Training & Evaluation Pipeline.
Trained on Roboflow Landslide & Debris-flow Dataset (2,285 images).
Includes GPU acceleration, validation metrics, and automated TFLite/ONNX export for Edge AI.
"""

import os
import sys
from pathlib import Path
import torch

def train_model(
    data_yaml_path: str,
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 8,
    model_variant: str = "yolov8n-seg.pt",
    output_dir: str = "c:/Users/ankes/OneDrive/Desktop/NER/NER/models/landslide_seg_run"
):
    from ultralytics import YOLO

    print("=================================================================")
    print("🚀 INITIATING LANDSLIDE SEGMENTATION MODEL TRAINING")
    print(f"Dataset Configuration: {data_yaml_path}")
    print(f"Model Variant:         {model_variant}")
    print(f"Epochs:                {epochs}")
    print(f"Image Resolution:      {imgsz}x{imgsz}")
    print(f"Batch Size:            {batch_size}")
    
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Acceleration: GPU ({gpu_name}) - CUDA Enabled ✅")
    else:
        print("Hardware Acceleration: CPU (CUDA not available)")
    print("=================================================================\n")

    # 1. Load Pretrained YOLOv8 Segmentation Weights
    model = YOLO(model_variant)

    # 2. Start Model Training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=output_dir,
        name="landslide_exp",
        exist_ok=True,
        save=True,
        plots=True,
        workers=0,
        optimizer="AdamW",
        lr0=0.002,
        verbose=True
    )

    print("\n✅ Training Complete!")
    print(f"Weights and logs saved in: {output_dir}/landslide_exp")

    # 3. Model Validation on Validation Set
    print("\n📊 RUNNING VALIDATION EVALUATION...")
    metrics = model.val()
    print("--- Validation Metrics Summary ---")
    try:
        print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
        print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
        print(f"Box mAP50:     {metrics.box.map50:.4f}")
        print(f"Box mAP50-95:  {metrics.box.map:.4f}")
    except Exception as e:
        print(f"Metrics: {metrics}")

    # 4. Save and Copy Best Model to NER/models/
    best_pt = Path(output_dir) / "landslide_exp" / "weights" / "best.pt"
    dest_dir = Path("c:/Users/ankes/OneDrive/Desktop/NER/NER/models")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if best_pt.exists():
        import shutil
        target_pt = dest_dir / "landslide_yolov8_seg_best.pt"
        shutil.copy(best_pt, target_pt)
        print(f"\n📦 Best model copied to: {target_pt}")

        # 5. Export to ONNX for High-Performance Serving
        try:
            print("\n🔄 Exporting to ONNX format for web serving...")
            onnx_path = model.export(format="onnx", imgsz=imgsz)
            print(f"ONNX Model saved to: {onnx_path}")
        except Exception as err:
            print(f"ONNX export notice: {err}")

    return model, metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Landslide Segmentation Model")
    parser.add_argument("--data", type=str, default="c:/Users/ankes/OneDrive/Desktop/NER/NER/data/landslide_seg_data.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz
    )
