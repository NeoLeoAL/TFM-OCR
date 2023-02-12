from tkinter import *
from tkinter.ttk import Style
from tkinterdnd2 import *

root = Tk()

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

root.config(bg='#fcb103') # BACKGROUND COLOR
root.resizable(False, False)
#root.iconbitmap('./assets/pythontutorial.ico')

style = Style()
print(style.theme_names())
print(style.theme_use())

print(style.element_names())

# Execute Tkinter
root.mainloop()