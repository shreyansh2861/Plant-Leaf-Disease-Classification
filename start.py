import sys
import os
from tkinter import *
import subprocess

window=Tk()

window.title("Choose a crop to classify......")
window.geometry('500x200')





def bell():
    os.system('python bell_pepper_test.py')

def rice():
    os.system('python rice_test.py')

def potato():
    os.system('python potato_test.py')



btn = Button(window, text="Bell Pepper", bg="black", fg="white",command=bell)
btn.grid(padx=230, pady=15)

btn = Button(window, text="Rice", bg="black", fg="white",command=rice)
btn.grid(padx=230, pady=15)

btn = Button(window, text="Potato", bg="black", fg="white",command=potato)
btn.grid(padx=230, pady=15)




window.mainloop()