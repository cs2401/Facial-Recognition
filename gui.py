from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from temp import *
import time

global count

count = 0


# The gui switches back and forth from these 2 functions every 2 seconds
def task1():
    C.itemconfig(image2, image=lst[2])

    print('hello')

    gui.after(2000, task2)

def task2():
    C.itemconfig(image2, image=lst[3])

    print('hello')

    gui.after(2000, task1)


# Creates the Gui object model
# We add widgets to the gui object
gui = Tk()


# Creates the canvas being used
C = Canvas(gui, height=200, width=320)

#filename = PhotoImage(file = "peppers.png")


#arr1 = Image.fromarray(arr('1.pgm'))
#arr1 = arr1.resize((115, 140), Image.ANTIALIAS)
#img1 =  ImageTk.PhotoImage(arr1)

#arr3 = Image.fromarray(arr('3.pgm'))
#arr3 = arr3.resize((115, 140), Image.ANTIALIAS)
#img3 =  ImageTk.PhotoImage(arr3)

#arr2 = Image.fromarray(arr('2.pgm'))
#arr2 = arr2.resize((115, 140), Image.ANTIALIAS)
#img2 =  ImageTk.PhotoImage(arr2)

test_set = []
for i in range(4):
    string = str(i+1) + '.pgm'
    arr1 = Image.fromarray(arr(string))
    arr1 = arr1.resize((115, 140), Image.ANTIALIAS)
    img1 =  ImageTk.PhotoImage(arr1)
    lst.append(img1)


image1 = C.create_image(30, 20, anchor=NW, image=lst[0])

image2 = C.create_image(175, 20, anchor=NW, image=lst[1])


C.pack()

C.create_text(90, 10, text="Looking For ....", font="Times 10 italic bold")

C.create_text(230, 10, text="Found!", font="Times 10 italic bold")


#B1 = Button(gui, text='Start', width=25, command=printyeet)

#B1.pack()

B2 = Button(gui, text='Stop', width=25, command=gui.destroy)

B2.pack()


gui.after(2000, task1)
# Once all widgets are added the application is launched by this infinite loop
# The loop continues until the application is closed
gui.mainloop()
