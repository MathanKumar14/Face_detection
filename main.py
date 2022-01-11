import cv2
import requests
import numpy as np
import imutils
# Load the cascade


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
url = "http://192.168.43.1:8080/shot.jpg"

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img1 = cv2.imdecode(img_arr, -1)
    img1 = imutils.resize(img1, width=1000, height=1800)
    # Read the frame
    #_, img = img1.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    light = 0
    try:
        if faces.all():
            light = 1
    except:
        light = 0

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img1)
    # Stop if escape key is pressed

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cv2.destroyAllWindows()