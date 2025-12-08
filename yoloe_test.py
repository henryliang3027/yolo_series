import os
from ultralytics import YOLOE

# specify image path
images_dir = '../vlm_20251030/training_data/images'
image_path = os.path.join(images_dir, '101.jpg')

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf.pt")

# Run prediction. No prompts required.
results = model.predict(image_path, conf=0.6)

print(f"Detected {len(results[0].boxes)} objects with high confidence.")

for box in results[0].boxes:
    x1,y1,x2,y2 = box.xyxy[0].int().tolist()
    # Get the confidence score
    confidence = box.conf[0].item()

    # Get the class ID
    class_id = int(box.cls[0].item())

    # Get the class label using model.names
    label = model.names[class_id]
    print(f"label: {label}, confidence: {confidence}, bbox: ({x1}, {y1}), ({x2}, {y2})")


# Show results
results[0].save('output.jpg')