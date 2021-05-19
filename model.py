from sklearn.model_selection import train_test_split
import re
import numpy as np
from skimage.measure import block_reduce
from matplotlib import pyplot


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


def generate_data(split):
    images = np.empty((400, 644))
    targets = np.empty(400).astype(np.uint8)

    for j in range(40):
        for i in range(10):
            n = 10*j + i
            image = read_pgm(f"FaceDataset/s{j+1}/{i+1}.pgm", byteorder='<')
            images[n] = image.reshape(644)
            targets[n] = j

    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=split)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    hat_matrix = []
    for i in range(10):
        X_train_i = X_train[y_train == i]
        H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
        hat_matrix.append(H_i)

    return hat_matrix


def model_predict(q, hat_matrix):
    distances = []
    for i in range(10):
        H_i = hat_matrix[i]
        y_i = H_i @ q
        d_i = np.linalg.norm(q - y_i, 2)
        distances.append(d_i)

    return distances.index(min(distances))


def evaluate_model(hat_matrix, X_test, y_test):
    total_count = len(y_test)
    correct_count = 0
    print(X_test.shape)
    print(y_test.shape)
    for i in range(len(y_test)):
        q = X_test[i]
        predicted_class = model_predict(q, hat_matrix)
        if predicted_class == y_test[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    return accuracy


def main():
    X_train, X_test, y_train, y_test = generate_data(0.5)
    hat_matrix = train_model(X_train, y_train)
    accuracy = evaluate_model(hat_matrix, X_test, y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
