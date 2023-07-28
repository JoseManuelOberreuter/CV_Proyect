import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, 
                help="path to the input video")  # Cambio de --image a --video
ap.add_argument("-c", "--cascade", 
                default="haarcascade_frontalcatface_extended.xml",
                help="path to cat detector haar cascade")
args = vars(ap.parse_args())

# Load the cat detector Haar cascade
detector = cv2.CascadeClassifier(args["cascade"])

# Load the video file
video_path = args["video"]
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cat faces in the frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(65, 65))

    # Draw rectangles and labels around the detected cat faces
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Cat #{}".format(i + 1), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Show the frame with cat detections
    cv2.imshow("Cat detection", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
