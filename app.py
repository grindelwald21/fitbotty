import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn
import tensorflow as tf
import random
import json
import keras
from cv2 import cv2
import mediapipe as mp
import time
import PoseModule as pm
import numpy as np
import matplotlib.pyplot as plt

import os
import urllib.request
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = np.array(training)
output = np.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.load("./model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

@app.route("/workoutBot", methods=['GET'])
def beginChat():
    return jsonify({"responses": "Start talking with the bot , type quit to stop the chatbot"})

@app.route("/workoutBot/problem/<string:inp>", methods=['GET'])
def askProblem(inp):
    while True:
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag and tag == "bench press":
                responses = tg['responses']
                return jsonify({"responses": random.choice(responses)})
                break
            elif tg['tag'] == tag and tag == "curls":
                responses = tg['responses']
                return jsonify({"responses": random.choice(responses)})
                break
            elif tg['tag'] == tag and tag == "shoulder press":
                responses = tg['responses']
                return jsonify({"responses": random.choice(responses)})
                break; 
            elif tg['tag'] == tag:
                responses = tg['responses']
                return jsonify({"responses": random.choice(responses)})
remarques= []
advices= []

@app.route('/workoutBot/upload/<string:inp>', methods=['POST'])
def upload_video(inp):
    
    if 'file' not in request.files:
        return jsonify({"responses": "No file part"})       
    file = request.files['file']
    if file.filename == '':
        return jsonify({"responses": "No image selected for uploading"})
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        cap = cv2.VideoCapture("static/uploads/"+filename)
        returned_cap = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'FMP4'), 20.0, (1280,720))

        results = model.predict([bag_of_words(inp, words)])        
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            
            if tg['tag'] == tag and tag == "bench press":
                detector = pm.poseDetector()
                list_angles_1 =[]
                list_angles_2 =[]
                time_angles = []
                i = 0
                start_time = time.time()
                seconds = 7
                while True:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time > seconds:
                        break
                    success, img = cap.read()
                    img = cv2.resize(img, (1280, 720))
                    img = detector.findPose(img)
                    lmList = detector.findPosition(img, False)
                    if len(lmList) != 0:
                        hand_grip_angle = detector.findAngle(img, 15, 11, 12)
                        elbow_angle = detector.findAngle(img, 23, 11, 13)
                        i += 1
                        time_angles.append(i)
                        list_angles_1.append(hand_grip_angle)
                        list_angles_2.append(elbow_angle)

                    returned_cap.write(img)
                    cv2.imshow("image", img)
                    cv2.waitKey(1)
                returned_cap.release()
                plt.plot(time_angles, list_angles_1)

                angle1 = min(list_angles_1)
                angle2 = min(list_angles_2)
                grip_ok = False
                elbow_ok = False

                if angle1 > 130:
                    remarques.append("you might want to close your hands a little bit, your hand's grip should be close to 1.5 you shoulder width")

                elif angle1 < 110 :
                    remarques.append("you might want to open up your hands a little bit more, to target your chest muscles more than your triceps") 

                else:
                    grip_ok = True
                    
                if angle2 > 55 :
                    remarques.append("you might want to close your elbow a bit more, it will reduce a lot your risk of injury")

                else:
                    elbow_ok = True
                        
                if grip_ok and elbow_ok:
                    remarques.append("Your position is very good, keep up the good work.")
                
                advices.append("There are very important key points to remember when performing the bench press to ensure healthy shoulders and longevity. In fact, these key points apply to the majority of all horizontal pressing movements.")
                advices.append("1. Keep a tight grip on the bar at all times, a tighter grip equates to more tension in the lower arms, upper back and chest.")
                advices.append("2. Keep your chest up (thoracic extension) throughout the movement.")
                advices.append("3. Elbows should be tucked and end up at approximately 45 degrees from your side")
                advices.append("4. Unrack the weight and take a deep breath and hold it.")
                advices.append("5. Row the weight down to your chest by pulling the bar apart like a bent over row. Do not relax and let the weight drop.")
                advices.append("6. Back, hips, glutes and legs are tight and isometrically contracted.")
                advices.append("7. When you touch your chest, drive your feet downward and reverse the movement.")
                advices.append("8. Lock out the elbows WITHOUT losing your arch and thoracic extension.")
                break
            elif tg['tag'] == tag and tag == "curls":
                               
                detector = pm.poseDetector()
                list_angles_1 =[]
                list_angles_2 =[]
                time_angles = []
                i = 0
                start_time = time.time()
                seconds = 7
                while True:
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    if elapsed_time > seconds:
                        break
                    
                    success, img = cap.read()
                    img = cv2.resize(img, (1280, 720))

                    img = detector.findPose(img)
                    lmList = detector.findPosition(img, False)
                    if len(lmList) != 0:
                        left_upper_arm_angle = detector.findAngle(img, 23, 11, 13)
                        right_upper_arm_angle = detector.findAngle(img, 14, 12, 24)
                        i += 1
                        time_angles.append(i)
                        list_angles_1.append(left_upper_arm_angle)
                        list_angles_2.append(right_upper_arm_angle)
                    
                    returned_cap.write(img)
                    cv2.imshow("image", img)
                    cv2.waitKey(1)
                    
                    
                returned_cap.release()
                plt.plot(time_angles, list_angles_1)

                angle1 = max(list_angles_1)
                angle2 = max(list_angles_2)
                hand1 = False
                Hand2 = False

                if angle1 > 20:
                    remarques.append("my advice is to keep your left upper arm straight and to move only your lower arm")
                else:
                    hand1 = True
                
                if angle2 > 20:
                    remarques.append("keep your right upper arm straight and to move only your lower arm")
                    
                else:
                    hand2 = True

                if hand1 and hand2:
                    remarques.append("Your position is very good, keep up the good work")
                advices.append("here is some general advices to get better and better:")
                advices.append("Ensure your elbows are close to your torso and your palms facing forward. Keeping your upper arms stationary, exhale as you curl the weights up to shoulder level while contracting your biceps.")
                advices.append("Use a thumb-less grip, advises Edgley. “Placing your thumb on the same side of the bar as your fingers increases peak contraction in the biceps at the top point of the movement,” he says. Hold the weight at shoulder height for a brief pause, then inhale as you slowly lower back to the start position.")
                break
            
            elif tg['tag'] == tag and tag == "shoulder press":
                                
                detector = pm.poseDetector()
                list_angles_1 =[]
                list_angles_2 =[]
                time_angles = []
                i = 0
                start_time = time.time()
                seconds = 7
                while True:
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    if elapsed_time > seconds:
                        break
                    
                    success, img = cap.read()
                    img = cv2.resize(img, (1280, 720))

                    img = detector.findPose(img)
                    lmList = detector.findPosition(img, False)
                    if len(lmList) != 0:
                        left_upper_arm_angle = detector.findAngle(img, 13, 11, 23)
                        right_upper_arm_angle = detector.findAngle(img, 24, 12, 14)
                        i += 1
                        time_angles.append(i)
                        list_angles_1.append(left_upper_arm_angle)
                        list_angles_2.append(right_upper_arm_angle)
                    
                    returned_cap.write(img)
                    cv2.imshow("image", img)
                    cv2.waitKey(1)
                returned_cap.release()    
                plt.plot(time_angles, list_angles_1)

                angle1 = min(list_angles_1)
                angle2 = min(list_angles_2)
                hand1 = False

                if angle1 < 80:
                    remarques.append("I can tell that your arms are down to much, try to keep your arms straight and don't let the bar down too much ")
                else:
                    hand1 = True
            
                if hand1:
                    remarques.append("Your position is very good, keep up the good work")
                advices.append("here is some general advices to get better and better:")
                advices.append("Don’t go overboard on the weight here, because this is an exercise that suddenly feels very tough halfway through a set. You almost want to feel like you’ve picked too light a weight for the first couple of reps.")
                advices.append("Opting for too heavy a weight can also mean you risk injury to your shoulders if your form gets sloppy as a result of the load.")
                advices.append("Hold the dumbbells by your shoulders with your palms facing forwards and your elbows out to the sides and bent at a 90° angle")
                advices.append("Without leaning back, extend through your elbows to press the weights above your head. Then slowly return to the starting position.")
                break;         
        return jsonify({"remarques": remarques, "advices": advices})
       

if __name__ == '__main__':
    app.run()