from tkinter import *
from tkinter import filedialog

def browseFiles():
    fileName = filedialog.askopenfilename(initialdir = "/", title = "Select a File", 
                                        filetypes = (("Image files", "*.jpg* *.png*"), ("all files", "*.*")))

    return fileName