import cv2

# Threshold to detect objects
thres = 0.45

# Initialize the video capture device
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width of the video capture (1280 pixels)
cap.set(4, 720)   # Set height of the video capture (720 pixels)
cap.set(10, 70)   # Set brightness of the video capture (70)

# Load the COCO class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained MobileNet V3 model for object detection
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Set input image size for the network
net.setInputScale(1.0 / 127.5)  # Normalize the input image by dividing by 127.5
net.setInputMean((127.5, 127.5, 127.5))  # Set the mean values for the input image
net.setInputSwapRB(True)  # Swap the red and blue channels in the input image

# Start the main loop to read frames from the video capture device, perform object detection, and display the results
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read from video capture device")
        break

    # Perform object detection on the input image
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if classIds is not None:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId > len(classNames):
                print("Error: Invalid classId: ", classId)
                continue

            # Draw a rectangle around the detected object
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Put text labels on the detected object
            label = f"{classNames[classId - 1].upper()} {round(confidence * 100, 2)}%"
            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Print the detected object's class name, confidence, and bounding box coordinates
            print(classNames[classId - 1], confidence, box)
    else:
        print("Error: No objects detected")

    # Show the output image with detected objects
    cv2.imshow("Output", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
