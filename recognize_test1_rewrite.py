import face_recognition
import os
import sys

known_labels = []
known_faces = []

for file in os.listdir(os.getcwd() + "/" + sys.argv[1] + "/learning"):
    face = face_recognition.load_image_file(os.getcwd() + "/" + sys.argv[1] + "/learning/" + file)
    label = file.split('.')[0]

    known_labels.append(label)
    known_faces.append(face_recognition.face_encodings(face)[0])


unknown_image = face_recognition.load_image_file(os.getcwd() + "/obama-trump_separated/test/obama2.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

for i in range(len(results)):
    print("Is it {}? - {}".format(known_labels[i], results[i]))
