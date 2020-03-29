from  tkinter import ttk,tix
from tkinter import *
from music import *
from emotionRecognitionSection import *
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
    label2=ttk.Label(root,text="Play Band Section",font=("Arial Bold", 10)).place(x=50,y=260)
    label3=ttk.Label(root,text="Actions: One Eye Open,Both Eyes Closed,Head on right,Head on Left,Head on Top",font=("Arial Bold", 8)).place(x=50,y=280)
    buttonPlay=ttk.Button(root, text="Play Band",style='C.TButton',command=Band).place(x=150,y=350)
    reset=ttk.Button(root,text="Reset",style='C.TButton',command=resetList).place(x=180,y=160)

    root.mainloop()#window constantly display

#main Execution
screen()
