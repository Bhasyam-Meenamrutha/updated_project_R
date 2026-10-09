import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

while True:

    # Capture one frame
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    # Display the frame
    cv2.imshow("Employee Attendance Camera", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()