from ultralytics import YOLO
import os

# Resume from last checkpoint if it exists, otherwise start fresh
checkpoint_path = '../runs/geometry_dash_detector_v3/weights/last.pt'

if os.path.exists(checkpoint_path):
    print(f"Resuming training from {checkpoint_path}")
    model = YOLO('checkpoint_path')
    resume = True
else:
    print("Starting fresh training")
    model = YOLO('yolo11s.pt')
    resume = False

model.train(
    data='../data/data.yaml',
    device="mps",
    
    # Training duration
    epochs=200,  # Train longer
    patience=100,  # Don't stop early
    resume=resume,  # Resume from checkpoint if available
    
    # Model size - REDUCED for memory
    imgsz=640,
    batch=4,  # Reduced from 8 to prevent OOM kills
    
    # CRITICAL FIXES for your issues:
    cls=2.0,  # Increase class loss weight - fixes "everything as background" problem
    conf=0.1,  # Lower confidence threshold - helps with low recall
    
    # Optimizer settings
    optimizer='AdamW',
    lr0=0.0005,  # Lower learning rate (was 0.001) - prevents overfitting
    lrf=0.01,   # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,  # More warmup epochs
    
    # Regularization to prevent overfitting
    dropout=0.1,  # Add dropout regularization
    
    # Data augmentation (helps with small dataset)
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,    # Saturation augmentation
    hsv_v=0.4,    # Value/brightness augmentation
    degrees=10,   # Rotation augmentation (±10 degrees)
    translate=0.1,  # Translation augmentation (10%)
    scale=0.5,    # Scale augmentation
    flipud=0.0,   # No vertical flip (game is horizontal)
    fliplr=0.5,   # Horizontal flip (50% chance)
    mosaic=0.3,   # Reduced mosaic (30% instead of 50%) - saves memory
    mixup=0.05,   # Reduced mixup (5% instead of 10%) - saves memory
    copy_paste=0.05,  # Reduced copy-paste (5% instead of 10%) - saves memory
    
    # Save frequently
    save_period=10,  # Save checkpoint every 10 epochs
    
    # Output
    project='/Users/kj16/Desktop/geometry-dash/runs',
    name='geometry_dash_detector_v3',
    save=True,
    plots=True,
    verbose=True
)