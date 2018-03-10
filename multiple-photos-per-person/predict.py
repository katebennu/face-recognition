import cv2
import os


def get_subjects_list(data_folder_path):
    subjects = [""]
    dirs = os.listdir(data_folder_path)

    for i in range(len(dirs)):
        dir_name = dirs[i]

        # ignore any non-relevant directories
        if dir_name.startswith("."):
            continue
        subjects.append(dir_name)
    return subjects


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def predict(face_recognizer, test_img, subjects):
    img = test_img.copy()
    face, rect = detect_face(img)
    # predict the image using our face recognizer
    prediction = face_recognizer.predict(face)

    print(subjects)
    print(prediction)

    label = prediction[0]
    return subjects[label]


def main():
    print("Loading model...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.read('savedModel.xml')

    print("Getting the list of subjects...")

    subjects = get_subjects_list("training-data")

    print("Predicting images...")

    target_image = cv2.imread("training-data/warren/w9.jpg")

    predicted_label = predict(face_recognizer, target_image, subjects)

    print("Prediction complete, face recognized: " + predicted_label)


main()