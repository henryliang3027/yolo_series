import cv2
from ultralytics import YOLO

# specify the model path
model_path = "finetuned_weights/yolov11s.pt"

# Initialize a YOLOE model
model = YOLO("yolov11s.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["bottle"]

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the current frame
    # The 'stream=True' argument is important for real-time processing
    results = model(frame, stream=True)

    # Process and display the results
    for r in results:
        # 'plot()' method draws bounding boxes, labels, and confidence scores directly on the frame
        annotated_frame = r.plot()
        cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()