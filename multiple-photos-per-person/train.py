import cv2
import os
import numpy as np

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = [""]

#function to detect face using OpenCV
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector
    #for a faster load replace with:
    #face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for i in range(len(dirs)):
        dir_name = dirs[i]

        # ignore any non-relevant directories if any
        if dir_name.startswith("."):
            continue
        subjects.append(dir_name)

        # ------STEP-2--------
        # labels must be integers
        label = i + 1

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            # cv2.imshow("Training on image...", image)
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)
            else:
                print('No face detected on the image: ' + image_name)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


def main():
    # let's first prepare our training data
    # data will be in two lists of same size
    # one list will contain all the faces
    # and the other list will contain respective labels for each face
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    print(labels)

    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    #create our LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #or use EigenFaceRecognizer by replacing above line with
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()

    #or use FisherFaceRecognizer by replacing above line with
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()

    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))

    face_recognizer.save('savedModel.xml')

    print(subjects)
main()