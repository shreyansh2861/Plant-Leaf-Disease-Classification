from tkinter import *

# loading Python Imaging Library
from PIL import ImageTk, Image

# To get the dialog box to open when required
from tkinter import filedialog


def open_img():
    # Select the Imagename from a folder
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=2)


def openfilename():

	# open file dialog box to select image
	# The dialogue box has a title "Open"
	filename = filedialog.askopenfilename(title ='SELECT IMAGE FOR TESTING')
	return filename

# Create a window
root = Tk()

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("550x300+300+150")

# Allow Window to be resizable
root.resizable(width = True, height = True)

# Create a button and place it into the window using grid layout
btn = Button(root, text ='open image', command = open_img).grid(
										row = 1, columnspan = 4)

root.mainloop()



# hello_psg.py

import PySimpleGUI as sg
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

img_pred = image.load_img(img, target_size=(150, 150))
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



