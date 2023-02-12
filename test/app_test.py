# https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
# https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/

from tkinter import *
from tkinter.ttk import *
from tkinterdnd2 import *
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk, ImageColor

def showImage(event):
    image1 = Image.open(event.data)    
    img = image1.resize((700, 300))
    test = ImageTk.PhotoImage(img)
    
    im = ImageColor.getcolor("gray", "L")
    print(im)

    label1 = Label(image=test)
    label1.image = test
    
    label1.drop_target_register(DND_FILES)
    label1.dnd_bind('<<Drop>>', showImage)
    
    rootWd = root.winfo_width()

    x = (rootWd / 2) - (img.width / 2)

    # Position image
    label1.place(x = x, y= 50)
    label1.pack()


# create root window
root = Tk()

style = ThemedStyle(root)
style.set_theme("xpnative")

# root window title and dimension
root.title("Optical character recognition")

window_width = 1024
window_height = 768

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

#root.config(bg='#fcb103') # BACKGROUND COLOR
root.resizable(False, False)
#root.iconbitmap('./assets/pythontutorial.ico')

# MENU FILE
menu = Menu(root)
subMenu = Menu(menu)
subMenu.add_command(label='New File')
subMenu.add_command(label='Save File')
subMenu.add_command(label='Settings')
subMenu.add_command(label='Exit')
menu.add_cascade(label='File', menu = subMenu)
root.config(menu = menu)

# IMAGEN DRAG AND DROP
image1 = Image.open('octocat.png')
img = image1.resize((700, 300))

test = ImageTk.PhotoImage(img)

label1 = Label(image=test)
label1.image = test

label1.drop_target_register(DND_FILES)
label1.dnd_bind('<<Drop>>', showImage)

root.update()
rootWd = root.winfo_width()

x = (rootWd / 2) - (img.width / 2)

# Position image
label1.place(x = x, y= 50)

# BOTON DE TRADUCCIÓN
b2 = Button(root,  width = 20, text = "TRADUCIR", command = root.destroy)

x2 = (rootWd / 2) - 80
y = img.height + 50 + 50
b2.place(x = x2, y = y)

# BOTON DE AUDIO
b3 = Button(root,  width = 5, text = "VOICE")

x2 = rootWd - 100
y = img.height + 50 + 70
b3.place(x = x2, y = y)

# TEXTAREA DE TEXTO TRADUCIDO
textarea = Text(root, height = 15, width = 120)
textarea.place(x = 30, y = y + 70)
textarea.config(state=DISABLED)

# Execute Tkinter
root.mainloop()
