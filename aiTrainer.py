from cv2 import cv2
import numpy as np
import time
import PoseModule as pm
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("C:/Users/DELL/Downloads/trainer/incline.mp4")
detector = pm.poseDetector()
list_angles_1 =[]
list_angles_2 =[]
time_angles = []
i = 0
start_time = time.time()
seconds = 4
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
        hand_grip_angle = detector.findAngle(img, 15, 11, 12)
        elbow_angle = detector.findAngle(img, 23, 11, 13)
        i += 1
        time_angles.append(i)
        list_angles_1.append(hand_grip_angle)
        list_angles_2.append(elbow_angle)
    
    cv2.imshow("image", img)
    cv2.waitKey(1)
    
def squat_depth_angle(body_parts, optimal_angle, thresh):
        ankle = average_or_one(body_parts, 10, 13)
        knee = average_or_one(body_parts, 9, 12)
        hip = average_or_one(body_parts, 8, 11)
        try:
            if ankle and knee and hip:
                angle_detected = calculate_angle(ankle, knee, hip)
            else:
                 return -1
        except TypeError as e:
            raise e
        # calculate percent deviation
        deviation = (optimal_angle - angle_detected)/optimal_angle * 100
        return deviation
    return squat_depth_angle(body_parts, (math.pi)/2, 0.1)