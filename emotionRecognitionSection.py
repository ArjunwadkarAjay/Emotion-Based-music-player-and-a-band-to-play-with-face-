from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras
from music import *
import matplotlib.pyplot as plt
from playsound import playsound

#variables for Emotion recognition model 
labels1={0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
labels2={0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Surprised', 4: 'Angry'}
label_img=[]

def recognizeStart():
    classifier = load_model('model_v6_23.hdf5')# Model1
    model = tensorflow.keras.models.load_model('keras_model.h5')#Model2
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _,img=cap.read()
        img=cv2.flip(img,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(img,1.1,5)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        #Model1
        for (x,y,w,h) in faces:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            if len(faces)==0:
                text="Sorry, No Face Detected"
                img=cv2.putText(img,text,(0,50),font,1,(0,0,255),1,cv2.LINE_AA)
            elif len(faces)>1:
                text="Multiple Faces Detected"
                img=cv2.putText(img,text,(0,50),font,1,(0,0,255),1,cv2.LINE_AA)
            else:
                #Model1
                roi1=img[y:y+h,x:x+w]
                roi1=cv2.resize(roi1,(48,48))
                gray1=cv2.cvtColor(roi1,cv2.COLOR_BGR2GRAY)
                roi1 = gray1.astype("float") / 255.0
                roi1= img_to_array(roi1)
                roi1 = np.expand_dims(roi1, axis=0)
                preds1 = (classifier.predict(roi1))[0]
                label_img.append(labels1[preds1.argmax()]) 
                #Model2
                roi2=img[y:y+h,x:x+w]
                size = (224, 224)
                roi2=cv2.resize(roi2,size)
                image_array = np.asarray(roi2)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                preds2=model.predict(data)[0]
                label_img.append(labels2[preds2.argmax()]) 
        cv2.imshow("image",img)
    cap.release()        
    cv2.destroyWindow("image")
    category=['Happy','Sad','Neutral']#dropping a few class for improving the music playing experience......
    large=0
    for i in category:
        if label_img.count(i)>=large:
            playlist=i
            large=label_img.count(i)
    print(playlist)
    playMusic(playlist)


def graph():
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    emotions = ['Happy','Sad','Neutral']
    counts = [label_img.count("Happy"),label_img.count("Sad"),label_img.count("Neutral")]
    ax.bar(emotions,counts)
    plt.show()
def resetList():
    label_img=[]
    
def Band():
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    cap=cv2.VideoCapture(0)
    while True:
        _,img=cap.read()
        img=cv2.flip(img,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("image",img)
        faces=face_cascade.detectMultiScale(img,1.1,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            roi_gray=gray[y:y+h,x:x+h]
            roi_color=img[y:y+h,x:x+h]
            eyes=eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
            if len(eyes)==1:
                playsound("Cymbal.wav")
            elif len(eyes)==0:
                playsound("Bass Drum.wav")
            else:
                pass
        if len(faces)==1:
            x,y,w,h=faces[0]
            width  =cap.get(3) 
            height =cap.get(4)
            if x<width/4:
                playsound("DRUM_ROL.wav")
            if x>3*width/4:
                playsound("Triangle.wav")
            if y<height/4:
                playsound("Cymbal.wav")
        
        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyWindow("image")

