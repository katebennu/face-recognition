import cv2
import os
import imghdr


def detect_face(image_path):
    #convert the test image to gray scale as opencv face detector expects gray images
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def find_images(dir_path):
    ''' finds files in a given path name and returns a list of cv images'''
    files = os.listdir(dir_path)
    images = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        if imghdr.what(file_path):
            images.append(file_path)
    return images


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def show_faces(original_img):
    img = original_img.copy()
    detect_face(img)
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


def main(path=".", show=False):
    images = find_images(path)
    for img in images:
        rect, face = detect_face(img)



if __name__ == "__main__":
    """arguments: first: relative directory path, second: show detected """
    import sys
    main(sys.argv[1], bool(sys.argv[2]))


# TODO:
# class Image with
# attributes: filename, face, rect, label,
# methods: detect face, draw rect, draw text, show image