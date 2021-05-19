from sklearn.model_selection import train_test_split
import re
import numpy as np
from skimage.measure import block_reduce


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
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
    image_downsampled = block_reduce(image, block_size=(4, 4), func=np.mean)
    return image_downsampled


def calculate_yi(q, i, X, y):
    X_train_i = X[y == i]
    H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
    y_i = H_i @ q
    d_i = np.linalg.norm(q - y_i, 2)
    return d_i


def generate_data(split, person_count=40, image_count_per_person=10):
    test_count_per_person = round(image_count_per_person * split)
    train_count_per_person = image_count_per_person - test_count_per_person

    # total_image_count = person_count * image_count_per_person

    X_train = np.empty((person_count*train_count_per_person, 644))
    y_train = np.empty((person_count*train_count_per_person))

    X_test = np.empty((person_count*test_count_per_person, 644))
    y_test = np.empty((person_count*test_count_per_person))

    # images = np.empty((400, 644))
    # targets = np.empty(400).astype(np.uint8)

    for j in range(person_count):
        images = np.empty((image_count_per_person, 644))
        targets = np.empty((image_count_per_person))
        for i in range(image_count_per_person):
            image = read_pgm(f"FaceDataset2/s{j+1}/{i+1}.pgm", byteorder='<')
            images[i] = image.reshape(644)

        person_X_train, person_X_test, person_y_train, person_y_test = train_test_split(images, targets,
                                                                                        test_size=split)

        for i in range(image_count_per_person):
            if i < train_count_per_person:
                n = train_count_per_person * j + i
                X_train[n] = person_X_train[i]
                y_train[n] = j
            else:
                n = test_count_per_person * j + (i - train_count_per_person)
                X_test[n] = person_X_test[i - train_count_per_person]
                y_test[n] = j

    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]

    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, person_count=40):
    hat_matrix = []
    for i in range(person_count):
        X_train_i = X_train[y_train == i]
        H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
        hat_matrix.append(H_i)

    return hat_matrix


def model_predict(q, hat_matrix, person_count=40):
    distances = []
    for i in range(person_count):
        H_i = hat_matrix[i]
        y_i = H_i @ q
        d_i = np.linalg.norm(q - y_i, 2)
        distances.append(d_i)

    return distances.index(min(distances))


def evaluate_model(hat_matrix, X_test, y_test, person_count):
    total_count = len(y_test)
    correct_count = 0
    for i in range(len(y_test)):
        q = X_test[i]
        predicted_class = model_predict(q, hat_matrix, person_count)
        if predicted_class == y_test[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    return accuracy


def main():
    person_count = 12
    image_per_person_count = 10
    X_train, X_test, y_train, y_test = generate_data(
        0.4, person_count, image_per_person_count)
    hat_matrix = train_model(X_train, y_train, person_count)

    accuracy = evaluate_model(hat_matrix, X_test, y_test, person_count)
    print(accuracy)


if __name__ == "__main__":
    main()
