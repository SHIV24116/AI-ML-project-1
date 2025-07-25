import cv2 as cv
import mediapipe as mp
import csv       # To store labeled data in csv
import os       #To check if csv file already exists 
import numpy as np

exercise_type = input("Enter the exercise name(e.g. squate,pushup,plank):").strip().lower()    #strip().lower()  removes extra spaces and converts to lower case

file_name= f"{exercise_type}_posture_dataset.csv"
if not os.path.isfile(file_name):
    with open(file_name,mode='w',newline='') as f:        #'mode = w' means file is available for writing   and 'r' means read only ##newline='' helps in avoiding extra spaces
        writer = csv.writer(f)
        writer.writerow(['back_angle','knee_angle','elbow_angle','label'])

#Initialize Mediapipe + open CV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing=mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

def angle_calcultor(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    radians =np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180/np.pi)
    if angle>180:
        angle = 360-angle
    return angle    

while cap.isOpened():
    ret,frame =cap.read()
    if not ret:
        break

    frame = cv.flip(frame,1)              # we flip frame horizontly to reverse mirror effect  (1 is for verticle axis)
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    result = pose.process(img)
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        shoulder  = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        wrist= [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        knee =  [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        #Angles
        back_angle = angle_calcultor(shoulder,hip,knee)
        knee_angle = angle_calcultor(hip,knee,ankle)
        elbow_angle= angle_calcultor(shoulder,elbow,wrist)

        cv.putText(img,f'Back Angle: {int(back_angle)}',(30,30),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv.LINE_AA)
        cv.putText(img,f'Elbow Angle: {int(elbow_angle)}',(30,80),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv.LINE_AA)
        cv.putText(img,f'Knee Angle: {int(knee_angle)}',(30,130),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv.LINE_AA)

    key = cv.waitKey(1) & 0xFF
    if key == ord('g'):
        with open(file_name,mode = 'a',newline = '')as f:             # mode='a'  creates file if not exists and appends i.e. writes at the end
            writer = csv.writer(f)
            writer.writerow([back_angle,knee_angle,elbow_angle,'Good'])
            print("Saved Good Posture")
    elif key == ord('b'):
        with open(file_name,mode ='a',newline ='') as f:
            writer = csv.writer(f)
            writer.writerow([back_angle,knee_angle,elbow_angle,'Bad'])
            print("Saved Bad Posture")

    img=cv.resize(img,(800,750))
    cv.imshow("Exercise Posture Collector",img) 
    if cv.waitKey(1) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()
                    



        




   
