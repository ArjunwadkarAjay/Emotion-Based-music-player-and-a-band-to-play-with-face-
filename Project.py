from  tkinter import ttk,tix
from tkinter import *

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras

import spotipy
import spotipy.util as util
import os
from random import randint

#variables for Emotion recognition model 
labels1={0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
labels2={0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Surprised', 4: 'Angry'}
label_img=[]
#variables for music player
username="dludfugf17jl42f9qvlbkjbnc"
scope = 'user-library-read'
client_id="eee90afdccc442fea8a0f2070b1fb507"
client_secret="53e0945705f24f429828825aef9fa52a"
redirect_uri="https://google.com"
happy=["alt-rock","ambient", "breakbeat","comedy","disco","disney","drum-and-bass","edm","electro", "electronic","happy","hard-rock"]
sad=["soul","study","sad","classical","death-metal","anime","black-metal"]
neutral=["dance","dancehall","dub","dubstep", "hardcore","hardstyle","holidays","honky-tonk","house","idm",]
dictSongsGenre={'Happy':happy,'Sad':sad,'Neutral':neutral}
def playMusic(song_type):
    try:
        token=util.prompt_for_user_token(username,scope,client_id=client_id,client_secret=client_secret,redirect_uri=redirect_uri)
    except:
        os.remove(".cache-{username}")
        token=util.prompt_for_user_token(username,scope,client_id=client_id,client_secret=client_secret,redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)
    #genre=dictSongsGenre[song_type][randint(0,len(dictSongsGenre[song_type]))]
    songList=sp.recommendations(seed_genres=dictSongsGenre[song_type])
    print(songList)
    
def recognizeStart():
    classifier = load_model('model_v6_23.hdf5')# Model1
    model = tensorflow.keras.models.load_model('keras_model.h5')#Model2
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _,img=cap.read()
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
    playMusic(playlist)

    
def graph():
    pass
def screen():
    root=Tk()#blank widow
    root.title("EMOTION BASED MUSIC PLAYER")
    root.geometry('500x500')
    style = ttk.Style()
    style.map("C.TButton",
        foreground=[('pressed', 'red'), ('active', 'blue')],
        background=[('pressed', '!disabled', 'black'), ('active', 'white')]
        )
    #entities on the window and placing them
    label1=ttk.Label(root,text="Emotion recognition section",font=("Arial Bold", 10)).place(x=50,y=30)
    buttonAnalyze=ttk.Button(root, text="Emotion Analyze",style='C.TButton',command=graph).place(x=90,y=110)
    buttonEmotionRecog=ttk.Button(root, text="Recognize Emotion",style='C.TButton',command=recognizeStart).place(x=250,y=110)
    #stop=ttk.Button(root, text="Stop",style='C.TButton',command=recognizeStop).place(x=400,y=110)
    label2=ttk.Label(root,text="Music Player Section",font=("Arial Bold", 10)).place(x=50,y=260)
    buttonPlay=ttk.Button(root, text="Play",style='C.TButton').place(x=50,y=350)
    buttonPause=ttk.Button(root, text="Pause",style='C.TButton').place(x=150,y=350)
    reset=ttk.Button(root,text="Reset",style='C.TButton').place(x=250,y=350)

    root.mainloop()#window constantly display

#main Execution
screen()










