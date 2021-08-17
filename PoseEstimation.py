from cv2 import cv2
import mediapipe as mp
import time
import PoseModule as pm
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("C:/Users/DELL/Downloads/trainer/incline.mp4")
detector = pm.poseDetector()
list_angles_1 =[]
list_angles_2 =[]
time_angles = []
exercice = input("what exercice you want to correct ?")
i = 0
start_time = time.time()
seconds = 7

if exercice in ["bench press", "incline press","incline bench press", "benching", "incline benching", "barbell bench press", "barbell incline", "barbell incline bench press"]:
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > seconds:
            break
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        #img = cv2.imread("C:/Users/DELL/Downloads/trainer/benching1.JPG")
        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            hand_grip_angle = detector.findAngle(img, 15, 11, 12)
            elbow_angle = detector.findAngle(img, 23, 11, 13)
            i += 1
            time_angles.append(i)
            list_angles_1.append(hand_grip_angle)
            list_angles_2.append(elbow_angle)

        cv2.imshow("image", img)
        cv2.waitKey(1)
    
    

    plt.plot(time_angles, list_angles_1)

    angle1 = min(list_angles_1)
    angle2 = min(list_angles_2)
    grip_ok = False
    elbow_ok = False


    if angle1 > 130:
        print("you might want to close your hands a little bit, your hand's grip should be close to 1.5 you shoulder width")

    elif angle1 < 110 :
        print("you might want to open up your hands a little bit more, to target your chest muscles more than your triceps") 

    else:
        grip_ok = True
        


    if angle2 > 55 :
        print("you might want to close your elbow a bit more, it will reduce a lot your risk of injury")

    else:
        elbow_ok = True
            

    if grip_ok and elbow_ok:
        print("Your position is very good, keep up the good work.")
    
    print("There are very important key points to remember when performing the bench press to ensure healthy shoulders and longevity. In fact, these key points apply to the majority of all horizontal pressing movements.")
    print("1. Keep a tight grip on the bar at all times, a tighter grip equates to more tension in the lower arms, upper back and chest.")
    print("2. Keep your chest up (thoracic extension) throughout the movement.")
    print("3. Elbows should be tucked and end up at approximately 45 degrees from your side")
    print("4. Unrack the weight and take a deep breath and hold it.")
    print("5. Row the weight down to your chest by pulling the bar apart like a bent over row. Do not relax and let the weight drop.")
    print("6. Back, hips, glutes and legs are tight and isometrically contracted.")
    print("7. When you touch your chest, drive your feet downward and reverse the movement.")
    print("8. Lock out the elbows WITHOUT losing your arch and thoracic extension.")    


elif exercice in ["curls", "bicep curls", "barbell picep curls", "dumbell bicep curls"]:
    list_angles_1 =[]
    list_angles_2 =[]
    time_angles = []
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > seconds:
            print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
            break
        
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        #img = cv2.imread("C:/Users/DELL/Downloads/trainer/benching1.JPG")
        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            left_upper_arm_angle = detector.findAngle(img, 23, 11, 13)
            right_upper_arm_angle = detector.findAngle(img, 14, 12, 24)
            i += 1
            time_angles.append(i)
            list_angles_1.append(left_upper_arm_angle)
            list_angles_2.append(right_upper_arm_angle)
        
        cv2.imshow("image", img)
        cv2.waitKey(1)
        
        

    plt.plot(time_angles, list_angles_1)

    angle1 = max(list_angles_1)
    angle2 = max(list_angles_2)
    hand1 = False
    Hand2 = False

    if angle1 > 20:
        print("my advice is to keep your left upper arm straight and to move only your lower arm")
    else:
         hand1 = True

    
    if angle2 > 20:
        print("keep your right upper arm straight and to move only your lower arm")
        
    else:
        hand2 = True
   

    if hand1 and hand2:
        print("Your position is very good, keep up the good work")
    print("here is some general advices to get better and better:")
    print("Ensure your elbows are close to your torso and your palms facing forward. Keeping your upper arms stationary, exhale as you curl the weights up to shoulder level while contracting your biceps.")
    print("Use a thumb-less grip, advises Edgley. “Placing your thumb on the same side of the bar as your fingers increases peak contraction in the biceps at the top point of the movement,” he says. Hold the weight at shoulder height for a brief pause, then inhale as you slowly lower back to the start position.")

elif exercice in ["shoulder press"] :
    list_angles_1 =[]
    list_angles_2 =[]
    time_angles = []
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > seconds:
            print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
            break
        
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        #img = cv2.imread("C:/Users/DELL/Downloads/trainer/benching1.JPG")
        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            left_upper_arm_angle = detector.findAngle(img, 23, 11, 13)
            right_upper_arm_angle = detector.findAngle(img, 14, 12, 24)
            i += 1
            time_angles.append(i)
            list_angles_1.append(left_upper_arm_angle)
            list_angles_2.append(right_upper_arm_angle)
        
        cv2.imshow("image", img)
        cv2.waitKey(1)
        
        

    plt.plot(time_angles, list_angles_1)

    angle1 = min(list_angles_1)
    angle2 = min(list_angles_2)
    hand1 = False

    if angle1 < 80:
        print("I can tell that your arms are down to much, try to keep your arms straight and don't let the bar down too much ")
    else:
         hand1 = True
   

    if hand1:
        print("Your position is very good, keep up the good work")
    print("here is some general advices to get better and better:")
    print("Ensure your elbows are close to your torso and your palms facing forward. Keeping your upper arms stationary, exhale as you curl the weights up to shoulder level while contracting your biceps.")
    print("Use a thumb-less grip, advises Edgley. “Placing your thumb on the same side of the bar as your fingers increases peak contraction in the biceps at the top point of the movement,” he says. Hold the weight at shoulder height for a brief pause, then inhale as you slowly lower back to the start position.")
    

else :
    print("yezi bla la3b")    
    
   







