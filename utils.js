import face_recognition
import os
import cv2

def load_known_faces(directory):
    encodings = []
    names = []

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            encodings.append(encoding[0])
            names.append(os.path.splitext(file)[0])

    return encodings, names

def recognize_faces(frame, known_encodings, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        names.append(name)

    return names
