from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from temp import *


# Creates the Gui object model
# We add widgets to the gui object
gui = Tk()


def printyeet():
    print('yeet')


C = Canvas(gui, height=200, width=320)

#filename = PhotoImage(file = "peppers.png")


arr1 = Image.fromarray(arr('1.pgm'))

arr1 = arr1.resize((115, 140), Image.ANTIALIAS)

img1 =  ImageTk.PhotoImage(arr1)

image1 = C.create_image(30, 20, anchor=NW, image=img1)


arr2 = Image.fromarray(arr('2.pgm'))

arr2 = arr2.resize((115, 140), Image.ANTIALIAS)

img2 =  ImageTk.PhotoImage(arr2)

image2 = C.create_image(175, 20, anchor=NW, image=img2)


C.pack()

C.create_text(90, 10, text="Looking For ....", font="Times 10 italic bold")

C.create_text(230, 10, text="Found!", font="Times 10 italic bold")

B = Button(gui, text='Stop', width=25, command=printyeet)

B.pack()

# Once all widgets are added the application is launched by this infinite loop
# The loop continues until the application is closed
gui.mainloop()
