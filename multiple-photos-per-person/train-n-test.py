import cv2
import os
import numpy as np
import math
import random

# float, defines the split between training and test data, 
# recommended between 0.6 and 0.9
TRAIN_FILES_PROPORTION = 0.8


def filter_hidden(subject_dir_path):
    return [f for f in sorted(os.listdir(subject_dir_path)) if not f.startswith(".")]

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


def prepare_data(data_folder_path):
    subjects = [""]
    dirs = filter_hidden(data_folder_path)

    faces = []
    labels = []
    files = []

    for i in range(len(dirs)):
        dir_name = dirs[i]

        # ignore any non-relevant directories
        if dir_name.startswith("."):
            continue
        subjects.append(dir_name)

        # labels must be integers
        label = i

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_train_images = filter_hidden(subject_dir_path)

        print(subject_train_images)

        for image_name in subject_train_images:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            # detect face
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
                files.append(image_name)
            else:
                print('No face detected on the image: ' + image_name)

    print(subjects)
    return faces, labels, files, subjects


def split_data(faces, labels, files):
    m = len(labels)
    train = math.floor(m * TRAIN_FILES_PROPORTION)
    test = m - train
    rand_numbers = [i for i in range(m)]
    random.shuffle(rand_numbers)
    shuffled_faces = [faces[i] for i in rand_numbers]
    shuffled_labels = [labels[i] for i in rand_numbers]
    shuffled_files = [files[i] for i in rand_numbers]
    return {
        'training_faces': shuffled_faces[: train],
        'training_labels': shuffled_labels[: train], 
        'training_files': shuffled_files[: train],
        'test_faces': shuffled_faces[-test :], 
        'test_labels': shuffled_labels[-test :],
        'test_files': shuffled_files[-test :]
    }


def predict(test_recognizer, faces, subjects):
    predictions = []
    for face in faces:
        # img = face.copy()

        # predict the image using our face recognizer
        prediction = test_recognizer.predict(face)

        print(subjects)
        print(prediction)

        label = prediction[0]
        predictions.append(label)
    return predictions


def evaluate_predictions(predictions, labels, subjects, files):
    m = len(predictions)
    correct = 0
    for i in range(m):
        print("Actual label: {}, predicted label: {}".format(labels[i], predictions[i]))
        if labels[i] == predictions[i]:
            correct += 1
        else:
            print("Incorrect prediction for file " + files[i])
    print("{} out of {} correct".format(correct, m))


def main():
    # 1. Get all files, recognize faces, return array of faces and array of labels
    print("Loading training data...")
    faces, labels, files, subjects = prepare_data("data")
    print(labels)
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    # 2. Split data into training and test sets
    split = {}
    split = split_data(faces, labels, files)
    print(split['test_labels']) 
    print(split['test_files'])

    # 3. Train model
    train_recognizer = cv2.face.LBPHFaceRecognizer_create()
    train_recognizer.train(split['training_faces'], np.array(split['training_labels']))
    train_recognizer.save('savedModel.xml')
    print('Training accomplished successfuly.')

    # 4. Test model
    input('Test the model?\n')
    print("Loading model...")
    test_recognizer = cv2.face.LBPHFaceRecognizer_create()
    test_recognizer.read('savedModel.xml')

    predicted_labels = predict(test_recognizer, split['test_faces'], subjects)

    evaluate_predictions(predicted_labels, split['test_labels'], subjects, split['test_files'])

main()