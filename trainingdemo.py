import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "datasets"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePath in imagePath:
        faceImage = cv2.imread(imagePath)
        faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceImage)
        ids.append(Id)
        cv2.imshow("Training", faceImage)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed!!")
