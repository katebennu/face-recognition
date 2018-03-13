import cv2

class Image:
    def __init__(self, file_path):
        self.path = file_path
        self.img_matrix = None
        self.img_matrix_gray= None
        self.face_matrix = None
        self.copy = None
        self.rect = None
        self.label = None

    def get_img_matrix(self):
        self.img_matrix = cv2.imread(self.path)
        self.img_matrix_gray= cv2.cvtColor(self.img_matrix, cv2.COLOR_BGR2GRAY)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.img_matrix_gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            return None, None
        (x, y, w, h) = faces[0]
        self.face_matrix = self.img_matrix_gray[y:y + w, x:x + h]
        self.rect = faces[0]

    def draw_rect(self, img):
        (x, y, w, h) = self.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_text(self, text, x, y):
        cv2.putText(self.img_matrix_gray, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def show_image(self):

        if not self.img_matrix:
            self.get_img_matrix()
        if not self.face_matrix or not self.rect:
            self.detect_face()

        self.copy = self.img_matrix.copy()
        # self.detect_face(img)
        self.draw_rect(self.copy)
        cv2.imshow('image', self.copy)
        # if label exists:
        # draw_text(img, label_text, rect[0], rect[1] - 5)
