import cv2 
import sys

id = sys.argv[1]

video=cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count=0

while True:
    ret,frame=video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+".jpg", frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(1)
    if k==ord("q"):
        break

    if count>550:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done..")
