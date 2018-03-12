import cv2
import os
import imghdr


def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def detect_faces(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if imghdr.what(file_path):
            image = cv2.imread(file_path)
            face, rect = detect_face(image)
            print(rect)