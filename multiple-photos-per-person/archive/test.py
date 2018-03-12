import cv2
import os
import math


# float between 0 and 1
TEST_FILES_PROPORTION = 0.3


def get_test_files(subject_dir_path):
    subject_files = [f for f in sorted(os.listdir(subject_dir_path)) if not f.startswith(".")]
    return subject_files[-1 * math.floor(len(subject_files) * TEST_FILES_PROPORTION):]


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


def prepare_test_data(data_folder_path):
    subjects = [""]
    faces = []
    labels = []

    dirs = os.listdir(data_folder_path)

    for i in range(len(dirs)):
        dir_name = dirs[i]

        # ignore any non-relevant directories
        if dir_name.startswith("."):
            continue
        subjects.append(dir_name)

        # labels must be integers
        label = i

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_train_images = get_test_files(subject_dir_path)

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
    print(labels, len(labels), len(faces))
    return faces, labels, subjects


def predict(face_recognizer, faces, subjects):
    predictions = []
    for face in faces:
        # img = face.copy()

        # predict the image using our face recognizer
        prediction = face_recognizer.predict(face)

        print(subjects)
        print(prediction)

        label = prediction[0]
        predictions.append(label)
    return predictions


def evaluate_predictions(predictions, labels, subjects):
    m = len(predictions)
    correct = 0
    for i in range(m):
        if labels[i] == predictions[i]:
            correct += 1
        print("Actual label: {}, predicted label: {}".format(labels[i], predictions[i]))
    print("{} out of {} correct".format(correct, m))


def main():
    print("Loading model...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('savedModel.xml')

    print("Loading test data...")
    faces, labels, subjects = prepare_test_data("data")

    print("Predicting images...")

    predicted_labels = predict(face_recognizer, faces, subjects)

    evaluate_predictions(predicted_labels, labels, subjects)
    # print("Prediction complete, face recognized: " + predicted_label)


main()