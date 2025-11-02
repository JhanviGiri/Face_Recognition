from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os


DETECTOR_PATH = "face_detection_model"   # folder containing deploy.prototxt and res10.caffemodel
EMBEDDING_MODEL_PATH = "openface_nn4.small2.v1.t7"
RECOGNIZER_PATH = "output/PyPower_recognizer.pickle"
LABEL_ENCODER_PATH = "output/PyPower_label.pickle"
CONFIDENCE_THRESHOLD = 0.7


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([DETECTOR_PATH, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR_PATH, "res10.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(RECOGNIZER_PATH, "rb").read())
le = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # choose color: green if >60%, red otherwise
            color = (0, 255, 0) if proba > 0.6 else (0, 0, 255)

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    fps.update()
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
