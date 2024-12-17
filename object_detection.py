import cv2 
import numpy
from picamera2 import Picamera2
from hough_lane_detector import HoughDetector

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1640, 1232)})
picam2.configure(config)
picam2.start()

try:
    while True:
        # capture_array returns the next frame as a NumPy array
        frame = picam2.capture_array()

        image = cv2.flip(frame, 0)
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        my_detector = HoughDetector()
        output_image = my_detector.process_image(image)
        X, Y = my_detector.projection_2_ground()
        print("the coordinates of target is ", X, Y)
        # Display the frame using OpenCV
        # cv2.imwrite("5.png", image)
        cv2.imshow("Camera Preview", output_image)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Clean up resources
    cv2.destroyAllWindows()
    picam2.stop()

    
'''
# Define the codec and create a VideoWriter object
# 'XVID' is a common codec. Other options: 'MJPG', 'MP4V', etc.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1640, 1232))

print("Press 'q' to stop recording...")

while True:
    frame = picam2.capture_array()

    image = cv2.flip(frame, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Write the frame to the output file
    out.write(image)

    # Display the frame
    cv2.imshow('Recording', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the windows
out.release()
cv2.destroyAllWindows()
picam2.stop()
'''
