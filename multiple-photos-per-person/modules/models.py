import cv2

class ImageFile:
    def __init__(self, file_path):
        self.path = file_path
        self.img_matrix = None
        self.face_matrix = None
        self.rect = None
        self.label = None

    def detect_face(self):
        # convert the test image to gray scale as opencv face detector expects gray images
        img = cv2.imread(self.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            return None, None
        # under the assumption that there will be only one face, extract the face area
        (x, y, w, h) = faces[0]
        # return only the face part of the image
        return gray[y:y + w, x:x + h], faces[0]
        pass

    def draw_rect(self):
        pass

    def draw_text(self):
        pass

    def show_image(self):
        pass
