# Facial-Recognition

Facial Recognition Software for Computer Vision (CITS4402)

This program aims to automatically process and downsample images to recognize and classify faces within a given face dataset. Each class contains 10 instances which will be partitioned according to the split specified (i.e. 70% used for training and 30% used for testing) The Linear Regression Classifcation (LRC) algorithm was utilised to train the model which works by computing the minimum distance between two given images. We provide accurate prediction for the face recognition by finding lowest distance between the different class images.

### Authors - Group 33

Benjamin Podmore (22504617)

Chintan Shah (22497366)

Ahbar Sakib (22512321)

### Version

v1.19.05.2021

### Directory Contents
	a. Data - contains each of the folders for the classes

	b. An executable to run the GUI
	
    c. Readme File which contains how to run the program
		i. Containing how to download the extra modules and packages

	d. Possibly an environment file (eg. Yaml file) if we are planning to use conda

### Using the software

*INSERT EXAMPLE OF FINAL GUI DISPLAY*

*Explain how to install the python packages and run it in a virtual environment with those dependencies*

### Procedure

*Test downsampling and different partioning splits of the dataset*

*Provide the accuracy results for each of these splits*

### Evaluation

#### Image Data Partioning:

The dataset requires a higher partioning/split for the training phase due to the lack of instances in the entire dataset. Since there are a total of 400 instances, it is better to use a 70/30 split (70% for training and 30% testing) and can be evaluated to be the optimal partioning.

There is also a risk of overfitting the data which is caused by using a very high percetage of the dataset for the training phase.

#### Downsampling

Downsampling the data helps with the time to train the model and make predictions efficiently. We have used a function from the Sci-kit Image library which scales down the images by a scale factor of 16. This factor can also be investigated as downsampling by a higher scale factor can produce inaccurate results due to the large aggregation of image pixel values.



