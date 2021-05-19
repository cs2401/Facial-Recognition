from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from temp import *
import time
from main import *
import random



# The fuction picks a random test image and proceeds to task2
def task1():

    # Selects a random test image to find using the linear regression
    # Updates the current image variable used
    rand = random.randint(0, (len(test_images) - 1))
    global current_image
    current_image = test_images[rand]

    # Changes the image1 (left) to this test image
    C.itemconfig(image1, image=current_image)

    # Goes to task2 after 2 seconds
    gui.after(2000, task2)


# The function cycles through a random test images and proceeds to task 3
count = 0
def task2():

    global count

    # If the function has completed 5 loop it moves to task 3
    if count == 5:

        # Updates the image2 (right) to a random image
        rand = random.randint(0, (len(train_images) - 1))

        C.itemconfig(image2, image=train_images[rand])

        gui.after(100, task1)

        count = 0
    
    # Cycles through 5 random images before predicting the actual match.
    else:

        # Updates the image2 (right) to a random image
        rand = random.randint(0, (len(train_images) - 1))

        C.itemconfig(image2, image=train_images[rand])

        gui.after(100, task2)

        count += 1



# Creates the Gui object model
# We add widgets to the gui object
gui = Tk()


# Creates the canvas being used
C = Canvas(gui, height=200, width=320)


X_train, y_train = train()
X_test, y_test = test()

# Given an np array of arrays of images as (644,) we want to create an list of the image objects
# We can use these to update the config for image1 and image2
def image_array(array):
    length = len(array)

    # Array of image objects used by the gui
    images_arr = np.empty((length), dtype=PhotoImage)

    # Goes through the length of the array to change all the arrays inside to image objects
    for i in range(length):

        # The internal arrays come in as (644,) we want them as (28,23)
        reshaped = array[i].reshape(28,23)

        # Adds to the array of the image objects
        arr1 = Image.fromarray(reshaped)
        arr1 = arr1.resize((115, 140), Image.ANTIALIAS)
        img1 =  ImageTk.PhotoImage(arr1)
        images_arr[i] = img1
    
    return images_arr

# Creates an image object array from the Training and Test set
train_images = image_array(X_train)
test_images = image_array(X_test)



# Creates the image objects.
# These will be updated by the tasks to display different images
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
