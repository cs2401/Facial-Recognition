from sklearn.model_selection import train_test_split
import re
import numpy as np
from skimage.measure import block_reduce



"""
Used the following code to read in the pgm files into a numpy array
source: https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
"""

def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    image = np.frombuffer(buffer,
                          dtype='u1' if int(
                                maxval) < 256 else byteorder+'u2',
                          count=int(width)*int(height),
                          offset=len(header)
                          ).reshape((int(height), int(width)))


    # Use blocK_reduce from scikit-image to downsample by factor of 4 along each axis                          
    image_downsampled = block_reduce(image, block_size=(4, 4), func=np.mean)
    return image_downsampled


"""
Generates X_train and X_test which hold the images, and y_train and y_test which hold the coresponding labels
Split is the fraction of images to become test data
Dataset=1 indicates given dataset, dataset=2 indicates dataset with new images
"""
def generate_data(split, dataset=1, person_count=40, image_count_per_person=10):
    test_count_per_person = round(image_count_per_person * split)
    train_count_per_person = image_count_per_person - test_count_per_person


    X_train = np.empty((person_count*train_count_per_person, 644))
    y_train = np.empty((person_count*train_count_per_person))

    X_test = np.empty((person_count*test_count_per_person, 644))
    y_test = np.empty((person_count*test_count_per_person))


    # Loop through each image
    for j in range(person_count):
        images = np.empty((image_count_per_person, 644))
        targets = np.empty((image_count_per_person))
        for i in range(image_count_per_person):
            image = read_pgm(
                f"FaceDataset{dataset}/s{j+1}/{i+1}.pgm", byteorder='<')
            images[i] = image.reshape(644)


        # For each person, randomly split and shuffle images into test and train images
        person_X_train, person_X_test, person_y_train, person_y_test = train_test_split(images, targets,
                                                                                        test_size=split)


        # Add the test and train images and labels for each person back into the whole test and train arrays
        for i in range(image_count_per_person):
            if i < train_count_per_person:
                n = train_count_per_person * j + i
                X_train[n] = person_X_train[i]
                y_train[n] = j
            else:
                n = test_count_per_person * j + (i - train_count_per_person)
                X_test[n] = person_X_test[i - train_count_per_person]
                y_test[n] = j


    # Shuffle training and test so all the images for each person aren't together

    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]

    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]

    return X_train, X_test, y_train, y_test



# """
# For a given image, q, a class, i, and training data X and y
# Find the l2 distance between q some sort of correlation with it belonging to class i
# """

# def calculate_yi(q, i, X, y):

#     # Retrieve all images belonging to class i
#     X_train_i = X[y == i]

#     # Calculate 
#     H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
#     y_i = H_i @ q
#     d_i = np.linalg.norm(q - y_i, 2)
#     return d_i

"""
Calculate the hat-matrix for the model which is analogus to weights in a linear regression
"""
def train_model(X_train, y_train, person_count=40):
    hat_matrix = []

    # Iterate through the people, or classes
    for i in range(person_count):

        # Retrieve all images belonging to class i
        X_train_i = X_train[y_train == i]

        # Calcualte weighting
        H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
        hat_matrix.append(H_i)

    return hat_matrix


"""
Given a image and a matrix of weights, classify the image
I.e. predict whose face it is
"""
def model_predict(q, hat_matrix, person_count=40):
    distances = []

    # Loop through each person
    for i in range(person_count):
        H_i = hat_matrix[i]
        y_i = H_i @ q

        # Calculate "distance"
        d_i = np.linalg.norm(q - y_i, 2)
        distances.append(d_i)

    # Classify as the people with the minimum distance
    return distances.index(min(distances))



"""
Calculate the accuracy of the model as number of correct prediction/total number of predictions
"""
def evaluate_model(hat_matrix, X_test, y_test, person_count):

    # Number of predictions to be made
    total_count = len(y_test)
    correct_count = 0
    for i in range(len(y_test)):
        q = X_test[i]

        # Get predicted class
        predicted_class = model_predict(q, hat_matrix, person_count)

        # Add a count to the number of correct predictions if the correct label is predicted
        if predicted_class == y_test[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    return accuracy


def main():
    person_count = 12
    image_per_person_count = 10
    test_split = 0.4
    dataset = 2

    X_train, X_test, y_train, y_test = generate_data(
        test_split, dataset, person_count, image_per_person_count)

    hat_matrix = train_model(X_train, y_train, person_count)

    accuracy = evaluate_model(hat_matrix, X_test, y_test, person_count)
    print(accuracy)


if __name__ == "__main__":
    main()
