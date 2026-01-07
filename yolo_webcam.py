import cv2
from ultralytics import YOLO
import math

# specify the model path
model_path = "finetuned_weights/best_4_class_20260107.pt"
# Initialize a YOLOE model
model = YOLO(model_path)  # or select yoloe-11s/m-seg.pt for different sizes


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

names = {0:"c0",1:"c1",2:"c2",3:"c3"}
colors = {0:(0, 255, 255),1:(41, 207, 0),2:(0, 161, 255),3:(255, 242, 0)}

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the current frame
    # The 'stream=True' argument is important for real-time processing
    results = model(frame, conf=0.6, stream=True)

    # Process and display the results

    r = list(results)[0]
    boxes = r.boxes.xyxy
    class_ids = r.boxes.cls
    confs = r.boxes.conf

    for box, class_id, conf in zip(boxes, class_ids, confs):
        # put box in cam
        x1,y1,x2,y2 = box.tolist()
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[label_id], 3)


        # object details
        org = [x1, y1 - 6]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6

        thickness = 2


        cv2.putText(frame, f'{names[label_id]} {conf:.2f}', org, font, fontScale, colors[label_id], thickness)

    # for r in results:
    #     # 'plot()' method draws bounding boxes, labels, and confidence scores directly on the frame
    #     annotated_frame = r.plot()
    #     cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    cv2.imshow("YOLO Webcam Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()