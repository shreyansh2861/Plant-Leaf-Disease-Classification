import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import model_from_json

#load the trained model to classify the images

from keras.models import load_model
json_file = open('tomato.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("tomato.h5")


classes = {
    0:'Tomato__Target_Spot',
    1: 'Tomato__Tomato_mosaic_virus',
    2:'Tomato__Tomato_YellowLeaf__Curl_Virus',
    3: 'Tomato_Bacterial_spot',
    4: 'Tomato_Early_blight',
    5:'Tomato_healthy',
    6:'Tomato_Late_blight',
    7:'Tomato_Leaf_Mold',
    8:'Tomato_Septoria_leaf_spot',
    9:'Tomato_Spider_mites_Two_spotted_spider_mite'
}
#initialise GUI

top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification Tomato')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((150,150))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = loaded_model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Choose an image",command=upload_image,
  padx=10,pady=5)

upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Tomato Leaf Disease Classification",pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()