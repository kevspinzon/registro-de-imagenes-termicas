from scipy import misc
from scipy import ndimage
from scipy import optimize
from skimage import feature

import matplotlib.pyplot as plt
import sys
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils
import scipy
from numba import vectorize, float64

theta = 350

rotation = np.array ([[np.cos(np.pi/theta),np.sin(np.pi/theta)],[-np.sin(np.pi/theta),np.cos(np.pi/theta)]])

# rotation = np.array ([[1,0],[0,1]])
scaling = np.array([-2,-2])
firstPoint = [[[113, 393],[84, 383],[86, 305],[200, 414],[191, 319],[230, 394],[258, 375],[279, 351],[289, 304],[240, 285]]


for i in range(11):
    firstPoint = np.dot(rotation,firstPoint) + scaling
    print (firstPoint)

# print (np.dot(rotation,firstPoint)+scaling)