import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
import os
import pandas as pd

import pyttsx3   # for text to speech
import time
import threading

 
last_tip = ""  # since we will be comparing it with the string so its better to use empty string
#last_tip = None   #to avoid repeating speech
last_spoken_time= 0

def speak_tip(tip):
    try:
        engine = pyttsx3.init()
        engine.say(tip)
        engine.runAndWait()
    except Exception as e:
        print("TTS error: ",e)    
 
exercise = input("Exrecise you are going to perform: ")
model_file = f"{exercise}_model.pkl"
encoderfile= f"{exercise}_label_encoder.pkl"

#Loading training dataset
df = pd.read_csv(f"{exercise}_posture_dataset.csv")

good_data = df[df["label"]=="Good"]
angle_stats = {
    "elbow": {
        "mean":
        good_data["elbow_angle"].mean(),
        "std":
        good_data["elbow_angle"].std()
    },
    "knee":{
        "mean":
        good_data["knee_angle"].mean(),
        "std":
        good_data["knee_angle"].std()
    },
    "back":{
        "mean":
        good_data["back_angle"].mean(),
        "std":
        good_data["back_angle"].std()
    }
}
def is_angle_correct(current_angle, angle_type):
    mean = angle_stats[angle_type]["mean"]
    std = angle_stats[angle_type]["std"]
    lower_bound = mean - 1.5 * std
    upper_bound = mean + 1.5 * std
    return lower_bound <= current_angle <= upper_bound

def mark_joint(img,landmark,color):
    cx,cy=int(landmark.x*img.shape[1]),int(landmark.y*img.shape[0])
    cv.circle(img,(cx,cy),10,color,-1)

if not os.path.isfile(model_file) or not os.path.isfile(encoderfile):
    print("Trained model or encoder not found for this exercise.")
    exit()

model = joblib.load(model_file)
labelencoder =joblib.load(encoderfile)    

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def angle_calculator(a,b,c):
    a = np.array(a)   #first point
    b = np.array(b)   #Mid point
    c = np.array(c)   #End point
    
     
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],c[0]-b[0])   # arctan2(dy,dx) calculates angle b/w vector(dy,dx) and X-axis

    angle = np.abs(radians*180/np.pi)
    if angle > 180.0:
        angle = 360-angle

    return angle

def validate_pose(landmarks, exercise_type):
    if exercise_type == "pushup":
        return is_in_pushup_pose(landmarks)
    #elif exercise_type == "squat":
    #    return is_in_squat_pose(landmarks)
    elif exercise_type == "plank":
       return is_in_plank_pose(landmarks)
    elif exercise_type == "standing":
        return is_in_standing_pose(landmarks)
    else:
        return True  # assume valid if no specific logic

def is_in_pushup_pose(landmarks):
    """
    landmarks: MediaPipe pose landmarks (normalized)
    Returns: True if in push-up position, False otherwise
    """

    # Get required landmark Y coordinates
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x

    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

    left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
    right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x

    # Take average for left and right side
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_knee_y = (left_knee_y + right_knee_y) / 2
    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
    avg_wrist_x = (left_wrist_x + right_wrist_x) / 2

    # Conditions (tuned for side view)
    back_aligned = abs(avg_shoulder_y - avg_hip_y) < 0.15 and abs(avg_hip_y - avg_knee_y) < 0.15 
    wrist_near_shoulder_level = abs(avg_shoulder_x - avg_wrist_x) < 0.10

    if back_aligned and wrist_near_shoulder_level:
        return True
    return False 

def is_in_standing_pose(landmarks):
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y


    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    hip_y = (left_hip_y + right_hip_y) / 2
    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    knee_y = (left_knee_y + right_knee_y) / 2

    vertical_posture =  ankle_y > knee_y > hip_y>shoulder_y  # Standing upright
    if vertical_posture and abs(hip_y - knee_y) > 0.15:
        return True
    return False

def is_in_plank_pose(landmarks):
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x

    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

    left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
    right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x

    # Take average for left and right side
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_knee_y = (left_knee_y + right_knee_y) / 2
    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
    avg_wrist_x = (left_wrist_x + right_wrist_x) / 2

    # Conditions (tuned for side view)
    back_aligned = abs(avg_shoulder_y - avg_hip_y) < 0.12 and abs(avg_hip_y - avg_knee_y) < 0.12 
    wrist_near_shoulder_level = abs(avg_shoulder_x - avg_wrist_x) < 0.10

    if back_aligned and wrist_near_shoulder_level:
        return True
    return False 


print("Starting WebCam...........\nPress d to exit")

# Web Cam Capturing......
capture = cv.VideoCapture(0)
while capture.isOpened:
    ret,frame = capture.read()
    if not ret:
        break

    #converting BGR frames to RGB to be supported for Mediapipe
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    result = pose.process(img)  #processes RGB frame to detect body landMarks

    #Drawing pose Landmarks
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)  #converting back to BGR for viewing in openCV
 
    if result.pose_landmarks:
      try:
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        # Validate pose before proceeding
        if validate_pose(landmarks, exercise):
            # Extract required keypoints
            shoulder  = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            wrist= [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            knee =  [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            

            # Angles
            back_angle = angle_calculator(shoulder, hip, knee)
            knee_angle = angle_calculator(hip, knee, ankle)
            elbow_angle = angle_calculator(shoulder, elbow, wrist)

            joint_angles={
                "elbow": elbow_angle,
                "knee": knee_angle,
                "back": back_angle
            }

            # Posture prediction
            x_input = [[back_angle, knee_angle, elbow_angle]]
            prediction = model.predict(x_input)[0]
            label = labelencoder.inverse_transform([prediction])[0]

            if label == "Bad":
                angle_tips = []

                for joint, angle in joint_angles.items():
                        if not is_angle_correct(angle, joint):
                            tip = f"Adjust your {joint.replace('_',' ')} angle to improve your posture"
                            angle_tips.append(tip)

                # Display tips on the frame
                for i, tip in enumerate(angle_tips):
                    cv.putText(img, tip, (30, 160 + i * 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

                # Speak only first tip if not recently spoken
                current_time= time.time()
                if angle_tips and angle_tips[0] != last_tip and (current_time - last_spoken_time > 10):
                    threading.Thread(target = speak_tip,args=(angle_tips[0],)).start()
                    last_tip = angle_tips[0]
                    last_spoken_time = current_time

             

            # 3. Draw skeleton color
            color = (0, 255, 0) if label == "Good" else (0, 0, 255)
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=color, thickness=2))
             

            # 4. Draw red/green dots on joints based on angle
            # (Place the joint coloring block here)
            joint_angle_mapping = {
                "elbow": {
                    "angle": elbow_angle,
                    "joints": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW],
                },
                "knee": {
                    "angle": knee_angle,
                    "joints": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
                },
                "back": {
                    "angle": back_angle,
                    "joints": [
                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_HIP,
                        mp_pose.PoseLandmark.RIGHT_HIP,
                        mp_pose.PoseLandmark.LEFT_KNEE,
                        mp_pose.PoseLandmark.RIGHT_KNEE,
                    ],
                }
            }

            for joint_name, data in joint_angle_mapping.items():
                angle_val = data["angle"]
                joints = data["joints"]

                color = (0, 255, 0) if is_angle_correct(angle_val, joint_name) else (0, 0, 255)

            for joint in joints:
                keypoint = landmarks[joint.value]
                cx, cy = int(keypoint.x * img.shape[1]), int(keypoint.y * img.shape[0])
                cv.circle(img, (cx, cy), 10, color, -1)

             
            
            # Displaying angles and result
            cv.putText(img, f"Posture detection for {exercise}", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(img, f'Back Angle: {int(back_angle)}', (30, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(img, f'Elbow Angle: {int(elbow_angle)}', (30, 80), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(img, f'Knee Angle: {int(knee_angle)}', (30, 100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(img, f"Posture: {label}", (30, 130), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0) if label == "Good" else (0, 0, 255), 2, cv.LINE_AA)


                        
        else:
            # Display message if user is not in the expected pose
            cv.putText(img, f"Not in {exercise} position", (30, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                        

      except IndexError:
        pass

    img = cv.resize(img,(800,700))  
    cv.imshow("Real Time Posture Detector",img)
    if cv.waitKey(2) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows
               
 

 