import cv2
import numpy as np

# Load YOLO model
weights_path = "yolov4-tiny.weights"  # Path to YOLO weights
config_path = "yolov4-tiny.cfg"       # Path to YOLO config
names_path = "coco.names"             # Path to class names

# Load YOLO network
net = cv2.dnn.readNet(weights_path, config_path)

# Load class names
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, np.ndarray) and unconnected_layers.ndim > 1:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]

# Assign random colors to each class for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open laptop camera (0 = default)
cap = cv2.VideoCapture(0)

# Function to detect objects in a frame
def detect_objects(frame):
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists to hold detection details
    class_ids = []
    confidences = []
    boxes = []

    # Analyze each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detection
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Max Suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Process the video feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    boxes, confidences, class_ids, indexes = detect_objects(frame)

    # Count vehicles
    vehicle_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in ["car", "bus", "truck", "motorbike"]:  # Vehicle types
                vehicle_count += 1
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Traffic Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

