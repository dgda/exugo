#!/usr/bin/env python
#python app.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
from imutils.video import VideoStream
from imutils.video import FPS
from firebase import firebase
import numpy as np
import argparse
import imutils
import datetime
import time
import cv2
from flask import Flask, render_template, Response

# emulated camera
from webcamvideostream import WebcamVideoStream

import cv2
firebase = firebase.FirebaseApplication('https://count-d7695.firebaseio.com/')



ap = argparse.ArgumentParser()

app = Flask(__name__, template_folder='')


totalCount = 0


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="weakest detection")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(len(CLASSES), 255, size=(len(CLASSES), 3))
print("Loading Model")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', "MobileNetSSD_deploy.caffemodel")



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
    


def gen(camera):
    """Video streaming generator function."""
    time.sleep(2.0)
    now = datetime.datetime.now()
    ten = now + datetime.timedelta(seconds=10)
    while True:
        frame = camera.read()
        now = datetime.datetime.now()

        (h, w) = frame.shape[:2]
        #gets blob from frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        
        net.setInput(blob)
        detected = net.forward()
        count = 0 
        for i in np.arange(0, detected.shape[2]):
            confidence = detected[0, 0, i, 2]
            if confidence > args["confidence"]:
                
                j = int(detected[0, 0, i, 1])
                box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[j],
                    confidence * 100)

                if(CLASSES[j] == "person"):
                    count += 1
                
                    rect = cv2.rectangle(frame, (startX, startY), (endX, endY),
                        COLORS[j], 2)
                    
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[j], 2)
            

            ret, jpeg = cv2.imencode('.jpg', frame)

            # print("after get_frame")
            if jpeg is not None:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                print("frame is none")

                
        if(now >= ten):
            ten = now + datetime.timedelta(seconds=5)
            result = firebase.post('CountedPeople', str(count) + " People As of " + str(datetime.datetime.now()))
            global totalCount
            totalCount = count
        print(count)

@app.route('/fetch')
def fetch():
    global totalCount
    return render_template('fetch.html', count=totalCount)
    

@app.route('/getDate')
def getDate():
    return render_template('getDate.html', date=str(datetime.datetime.now()))
    

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(WebcamVideoStream().start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', port=80, debug=True, threaded=True)