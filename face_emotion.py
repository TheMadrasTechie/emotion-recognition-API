import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import json
import datetime
import random
import string

def generate_random_name():
    # Get the current date and time
    now = datetime.datetime.now()

    # Format it as a string in the YYYYMMDDHHmmSS format
    date_string = now.strftime("%Y%m%d%H%M%S")

    # Generate a random string of 5 characters
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    # Combine the date string and the random string
    name = date_string + random_string

    return name



def detect_emotion(frame):
    # Initiate emotion detector
    emotion_detector = FER(mtcnn=True)

    # Detect emotions
    result = emotion_detector.detect_emotions(frame)
    if len(result) > 0:
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]

        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 71, 151), 2)
        emotion_name, score = emotion_detector.top_emotion(frame )
        emotion_values = list(emotions.values())
        emotion_keys = list(emotions.keys())
        max_emotion = emotion_keys[emotion_values.index(max(emotion_values))]
        tmp_name = "emotion_face//"+str(generate_random_name())+".jpg"
        cv2.imwrite(tmp_name,frame)
        return json.dumps({"emotion": max_emotion,"result":emotions})
    else:
        print("No Face Detected")
        return None

# image = cv2.imread('sm.png')
# emotion_json = detect_emotion(image)
# print(emotion_json)