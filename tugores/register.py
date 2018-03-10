import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils


def compare(image1, image2):
  return np.sum((image1 - image2) ** 2)

def register_point(image1, image2, point1, search_size = 100):
  w, h = point1
  half = search_size // 2
  point2 = 0, 0
  error2 = np.inf
  for i in range(w - half, w + half):
    for j in range(h - half, h + half):
      pointi = (i, j)
      thumb1 = utils.thumb(image1, point1)
      thumbi = utils.thumb(image2, pointi)
      errori = compare(thumb1, thumbi)
      if errori <= error2:
        point2 = pointi
        error2 = errori
  return point2

def register_points(image1, image2, points):
  return [register_point(image1, image2, pointi) for pointi in points]

def register(images):
  points = [utils.ask_points(images[0])] #pregunta por los puntos
  for i in range(1, len(images)): #desde 1 a n
    image1 = utils.read(images[i - 1]) #toma imagen n-1
    image2 = utils.read(images[i]) #toma imagen n
    point1 = points[i - 1] #toma los puntos de la imagen n-1
    point2 = register_points(image1, image2, point1) 
    points.append(point2)
  return points

if __name__ == '__main__':
  path = sys.argv[1]
  images = utils.images(path)
  points = register(images)
  utils.render_points(images, points)
