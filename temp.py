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


#array = read_pgm('1.pgm')
#print(array)
#print('\n')
#print(array.reshape(644))
#print('\n')
#print(array.reshape(28,23))

def arr(name):
    array = read_pgm(name)
    return array


#images = np.empty((100,644))
#targets = np.empty(100).astype(np.uint8)

#for j in range(10):
#    for i in range(10):
#        if i != 5 and i!= 6:
#            n = 10*j + i
#            image = read_pgm(f"s{j+1}\{i+1}.pgm", byteorder='<')
#            images[n] = image.reshape(644)
#            targets[n] = j