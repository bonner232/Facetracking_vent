# Import the necessary libraries:
import cv2
import numpy as np
import requests

# Set the URL for the live video stream:
# This URL is where the video stream is expected to be available.
url = 'http://192.168.1.65'

# Create a named window for displaying the video stream:
cv2.namedWindow("Live Transmission", cv2.WINDOW_AUTOSIZE)

# Prepare the stream URL and initiate the request for the video stream:
# The stream object is used to retrieve the video stream in chunks.
stream_url = url.rstrip('/') + "/"
stream = requests.get(stream_url, stream=True)

# Initialize variables for buffering the received data:
# These variables are used to handle the chunked data received from the stream.
bytes_buffer = bytes()
delimiter = b'\r\n--123456789000000000000987654321\r\n'

# Load the pre-trained Haar cascades for face and eye detection:
# These cascade classifiers are used for detecting faces and eyes in the video frames.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start processing the video stream, The stream is iterated in chunks of 8192 bytes.
for chunk in stream.iter_content(chunk_size=8192):

    # Buffer the received data and search for the delimiter to identify complete image frames:
    # The delimiter is used to separate individual image frames within the stream.
    bytes_buffer += chunk
    while True:
        start = bytes_buffer.find(delimiter)
        if start == -1:
            break
        end = bytes_buffer.find(delimiter, start + len(delimiter))
        if end == -1:
            break

        # Extract the JPEG image data from the buffered bytes:
        jpg = bytes_buffer[start + len(delimiter):end]
        bytes_buffer = bytes_buffer[end + len(delimiter):]

        # Skip any additional content in the image data:
        content_start = jpg.find(b'\r\n\r\n') + 4
        img_data = jpg[content_start:]

        # Decode the image data into an OpenCV image format:
        # The cv2.imdecode() function converts the image data from bytes to an OpenCV image format.
        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Process the image if it is valid:
        # The script converts the image to grayscale and uses the face cascade classifier to detect faces in the frame.
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Draw rectangles around the detected faces and eyes:
            # This code loops through the detected faces and eyes and draws rectangles around them.
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Display the processed frame in a window:
            cv2.imshow("Live Transmission", img)
        else:
            print("Received invalid image frame")

        # Check for user input to quit the program, Pressing the 'q' key terminates the program.
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# This code ensures that the window is closed after the program finishes.
cv2.destroyAllWindows()
