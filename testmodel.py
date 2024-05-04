import cv2 
import os
import sys

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

username = sys.argv[1] if len(sys.argv) > 1 else None  # Retrieve username from command-line arguments if provided

if username:
    with open("usernames.txt", "r+") as file:
        usernames = file.readlines()
        if username + '\n' not in usernames:
            file.write(username + "\n")
            username=False
        # else:
        #     print("Username already exists.")
        #     sys.exit()  # Exit the program if the username already exists

# Read usernames from the file and append them to the name_list
name_list = [""]
with open("usernames.txt", "r") as file:
    for line in file:
        name = line.strip()
        if name:
            name_list.append(name)
file_paths=[""]
for name in name_list:
    if name!="":
        file_paths.append(name+".txt")
    

# Create text files for each name in the name_list

if not os.path.exists(f"{username}.txt") and name!=" ":
    file_path =f"{username}.txt"
    with open(file_path, 'w') as file:
        file.write(f"This is {username}'s text file.")
    
file_opened = {}
print(file_paths)
print(name_list)

def confidence_to_accuracy(confidence):
    return 100 - confidence 

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        accuracy = confidence_to_accuracy(conf)
        print(accuracy)
        if accuracy >= 47:
            if serial < len(name_list):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, name_list[serial], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if serial not in file_opened or not file_opened[serial]:
                    if serial < len(file_paths):
                        file_path = file_paths[serial]
                        if os.path.exists(file_path):
                            os.startfile(file_path)
                            file_opened[serial] = True
                        else:
                            print(f"File for {name_list[serial]} not found.")
                    else:
                        print("Index out of range for file_paths.")
            else:
                print("Index out of range for name_list.")
            break
            
        else:
            # print(accuracy)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            # cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            # cv2.putText(frame,"unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if serial in file_opened and file_opened[serial]:
                video.release()
                cv2.destroyAllWindows()
                exit(0)

    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

