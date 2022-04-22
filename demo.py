from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename


# hello_psg.py
import glob
import io
import os
from tkinter import filedialog
from PIL import ImageTk
import tkinter

import PySimpleGUI as sg
from PIL import Image

from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# load json and create model
json_file = open('tomato.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("tomato.h5")

layout = [[sg.Text("Tomato testing model loaded, choose an image of tomato leaf")], [sg.Button("Choose file")]]

# Create the window
window = sg.Window("Demo", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Choose file":
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        window.close()
    elif event == sg.WIN_CLOSED:
        break








img_pred = image.load_img(filename, target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

result = loaded_model.predict(img_pred)
print(result)

if result[0][0] == 1:
    prediction = 'Target Spot'
elif result[0][1]:
    prediction = 'Mosaic Virus'
elif result[0][2]:
    prediction = 'Yellow leaf curl virus'
elif result[0][3]:
    prediction = 'Bacterial Spot'
elif result[0][4]:
    prediction = 'Early Blight'
elif result[0][5]:
    prediction = 'Healthy'
elif result[0][6]:
    prediction = 'Late Blight'
elif result[0][7]:
    prediction = 'Leaf Mold'
elif result[0][8]:
    prediction = 'Septoria Leaf Spot'
elif result[0][9]:
    prediction = 'Spider Mites Two Spotted Spider Mites'
print(prediction)


layout = [[sg.Text(prediction)], [sg.Button("OK")]]

# Create the window
window = sg.Window("Demo", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()
