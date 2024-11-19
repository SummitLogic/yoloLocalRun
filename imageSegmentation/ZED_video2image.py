import cv2
import numpy as np

# Create a VideoCapture object and read from the input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("WIN_20240131_11_30_37_Pro.mp4")

if cap.isOpened() == False:
    print("Error opening video stream or file")

#Obtener el primer frame
ret, prev_frame = cap.read()
if not ret:
    print("Error reading the first frame")
    exit()

# Read until the video is completed
i = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Compare the current frame with the previous frame
        diff = cv2.absdiff(prev_frame, frame)
        diff_percentage = np.count_nonzero(diff) / frame.size

        # Threshold
        threshold = 0.20  


        if diff_percentage >= threshold:
            (h, w) = frame.shape[:2]
            centerX, centerY = (w // 2), (h // 2)
            leftPart = frame[:, 0:centerX]
            rightPart = frame[:, centerX:]

            # imshow
            cv2.imshow("Left Part", leftPart)
            cv2.imshow("Right Part", rightPart)  # left y right son lo mismo
            cv2.imwrite("RoboBoatVideo_frame" + str(i) + ".jpg", leftPart)

            i += 1

            prev_frame = frame  # Update the previous frame

        # Press Q on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break
cap.release()

cv2.destroyAllWindows()