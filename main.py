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
    image_downsampled = block_reduce(image,block_size=(4,4),func=np.mean)
    return image_downsampled


def calculate_yi(q, i):
    X_train_i = X_train[y_train == i]
    H_i = X_train_i.T @ np.linalg.pinv(X_train_i @ X_train_i.T) @ X_train_i
    y_i = H_i @ q
    d_i = np.linalg.norm(q - y_i, 2)
    return d_i


images = np.empty((400,644))
targets = np.empty(400).astype(np.uint8)


for j in range(40):
    for i in range(10):
        n = 10*j + i
        image = read_pgm(f"FaceDataset\s{j+1}\{i+1}.pgm", byteorder='<')
        images[n] = image.reshape(644)
        targets[n] = j

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, targets,test_size=0.3)

def train():
    return X_train, y_train

def test():
    return X_test, y_test


q = X_test[y_test == 5][0]

pyplot.imshow(q.reshape(28,23), pyplot.cm.gray)
pyplot.show()



distances = []
for i in range(10):
    distances.append(calculate_yi(q, i))


print(distances)
print(distances.index(min(distances)))




    