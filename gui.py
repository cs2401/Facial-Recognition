from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from temp import *
import time
from main import *
import random



# The gui switches back and forth from these 2 functions every 2 seconds
def task1():

    rand = random.randint(0, (len(test_images) - 1))

    global current_image

    current_image = test_images[rand]

    C.itemconfig(image1, image=test_images[rand])

    gui.after(2000, task2)
    #print(1)

count = 0
def task2():

    global count

    if count == 5:
        rand = random.randint(0, (len(train_images) - 1))

        C.itemconfig(image2, image=train_images[rand])

        gui.after(100, task1)

        count = 0
        #print(2)
    
    else:
        rand = random.randint(0, (len(train_images) - 1))

        C.itemconfig(image2, image=train_images[rand])

        gui.after(100, task2)

        #print(3)

        count += 1


    


# Creates the Gui object model
# We add widgets to the gui object
gui = Tk()


# Creates the canvas being used
C = Canvas(gui, height=200, width=320)

#filename = PhotoImage(file = "peppers.png")


#arr1 = Image.fromarray(arr('1.pgm'))
#arr1 = arr1.resize((115, 140), Image.ANTIALIAS)
#img1 =  ImageTk.PhotoImage(arr1)
#print(type(img1))

#arr3 = Image.fromarray(arr('3.pgm'))
#arr3 = arr3.resize((115, 140), Image.ANTIALIAS)
#img3 =  ImageTk.PhotoImage(arr3)

#arr2 = Image.fromarray(arr('1.pgm'))
#arr2 = arr2.resize((115, 140), Image.ANTIALIAS)
#img2 =  ImageTk.PhotoImage(arr2)


X_train, y_train = train()
X_test, y_test = test()

# Given an np array of arrays of images as (644,) we want to create an list of the image objects
# We can use these to update the config for image1 and image2
def image_array(array):
    length = len(array)

    images_arr = np.empty((length), dtype=PhotoImage)

    for i in range(length):
        reshaped = array[i].reshape(28,23)
        arr1 = Image.fromarray(reshaped)
        arr1 = arr1.resize((115, 140), Image.ANTIALIAS)
        img1 =  ImageTk.PhotoImage(arr1)
        images_arr[i] = img1
    
    return images_arr

train_images = image_array(X_train)
test_images = image_array(X_test)




image1 = C.create_image(30, 20, anchor=NW, image=test_images[0])

image2 = C.create_image(175, 20, anchor=NW, image=train_images[0])

C.pack()


# Adds text to the Canvas 
C.create_text(90, 10, text="Looking For ....", font="Times 10 italic bold")

C.create_text(230, 10, text="Found!", font="Times 10 italic bold")


# Button that quits the application
B2 = Button(gui, text='Stop', width=25, command=gui.destroy)

B2.pack()


# This is a global variable shared between all the task function
# It is the current test image being inspected
current_image = test_images[0]


# Executes task1 after 2 seconds
gui.after(2000, task1)


# Once all widgets are added the application is launched by this infinite loop
# The loop continues until the application is closed
gui.mainloop()
