from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from face_emotion import detect_emotion  # assuming the function is imported from here
import json

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_as_json(input_string):
    # Load the input string as a Python dictionary
    input_dict = json.loads(input_string)
    return input_dict

class ImageType(BaseModel):
    url: str

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    emotion = detect_emotion(img)  # get the emotion
    emotion_value = format_as_json(emotion)
    print(emotion_value)

    #return {"prediction": emotion, "message": "Emotion detection completed"}
    return emotion_value
