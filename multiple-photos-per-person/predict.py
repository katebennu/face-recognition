import cv2
import os


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


def get_subjects_list(data_folder_path):
    subjects = [""]
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
    return subjects


# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# this function recognizes the person in image passed
def predict(face_recognizer, test_img, subjects):
    # make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    prediction = face_recognizer.predict(face)

    print(subjects)
    print(prediction)

    label = prediction[0]

    return subjects[label]


def main():
    print("Loading model...")
    #create our LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #or use EigenFaceRecognizer by replacing above line with
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()

    face_recognizer.read('savedModel.xml')

    print("Getting list of subjects...")

    subjects = get_subjects_list("training-data")

    print("Predicting images...")

    # load test images
    test_img1 = cv2.imread("test-data/test1.jpg")
    # test_img2 = cv2.imread("test-data/test2.jpg")

    # perform a prediction
    predicted_label = predict(face_recognizer, test_img1, subjects)
    # predicted_img2 = predict(test_img2)
    print("Prediction complete, face recognized: " + predicted_label)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()