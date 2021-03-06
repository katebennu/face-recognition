import cv2
import os
import numpy as np
import math

# float between 0 and 1
TRAIN_FILES_PROPORTION = 0.7


def get_train_files(subject_dir_path):
    subject_files = [f for f in sorted(os.listdir(subject_dir_path)) if not f.startswith(".")]
    return subject_files[: math.floor(len(subject_files) * TRAIN_FILES_PROPORTION)]


def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    subjects = [""]
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for i in range(len(dirs)):
        dir_name = dirs[i]

        # ignore any non-relevant directories
        if dir_name.startswith("."):
            continue
        subjects.append(dir_name)

        # labels must be integers
        label = i

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_train_images = get_train_files(subject_dir_path)

        print(subject_train_images)

        for image_name in subject_train_images:

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            # detect face
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                print('No face detected on the image: ' + image_name)

    print(subjects)
    return faces, labels


def main():
    print("Loading training data...")
    faces, labels = prepare_training_data("data")
    print("Data prepared")

    print(labels)

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save('savedModel.xml')


main()
