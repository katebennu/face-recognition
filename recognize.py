import face_recognition
import os
import sys

known_labels = []
known_faces = []

base_path = os.getcwd() + "/" + sys.argv[1]
learning_path = base_path + "/learning/"
test_path = base_path + "/test/"

for file in os.listdir(learning_path):
    face = face_recognition.load_image_file(learning_path + file)
    label = file.split('.')[0]

    known_labels.append(label)
    known_faces.append(face_recognition.face_encodings(face)[0])


unknown_image = face_recognition.load_image_file(test_path + os.listdir(test_path)[0])
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

for i in range(len(results)):
    print("Is it {}? - {}".format(known_labels[i], results[i]))
