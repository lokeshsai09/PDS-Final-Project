import cv2
import numpy as np
from keras.models import load_model
from MusicUtil import getMusicByMood

global df
df = getMusicByMood("Angry")

def Stream():
    #6241 images in sad data.
    emotion_dict = {0: "Angry", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy",5:"sadness",6:"surprise"}

    model =  load_model("model/CK+48.h5")
    print("Loaded CK+48 model from disk")
    
    cap = cv2.VideoCapture(1)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(frame, (48, 48)), 3), 0)

            # predict the emotions
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            global df
            df = getMusicByMood(emotion_dict[maxindex])
            print(emotion_dict[maxindex])
            #for i in range(5):
            #    print(df[i])
        #cv2.imshow('Emotion Detection', frame)
        out = cv2.imencode('.jpg',frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+out+b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def getSongs():
    return df