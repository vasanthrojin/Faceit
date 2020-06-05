import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from shutil import copyfile
import time

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("../../project repos/face_organiser/faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img)
    file_name = []
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding,0.50)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (11,106,176), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (11,106,176), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 0.4, (255, 255, 255), 2)

            if name != "Unknown" and name not in file_name:
                if name in os.listdir(os.getcwd()+"/people_classified"):
                    copyfile(im, "people_classified/"+name+"/"+name+str(time.time())+".jpg")
                    file_name.append(name)
                else:
                    os.makedirs("people_classified/"+name)
                    copyfile(im,"people_classified/"+name+"/"+name+str(time.time())+".jpg")
                    file_name.append(name)




    # Display the resulting image

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


for i in os.listdir(os.getcwd()+"/Unknown images"):
    print(classify_face("/Unknown images/"+i))
#
# print(classify_face("Unknown images/unknown3.jpg"))



