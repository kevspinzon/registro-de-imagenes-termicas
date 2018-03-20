from scipy import misc
from scipy import ndimage
from scipy import optimize
from skimage import feature


import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils
import scipy


#x= parametros de transformacion i=imagen original i2=imagen objetivo
def E(x,i,i2):
    #Solo para traslado en el eje X y Y
    #i3 = affin(i,1,0,0,1,x[0],x[1]) 

    # i3 = affin(i,math.cos(math.pi/x[2]),math.sin(math.pi/x[2]),-math.sin(math.pi/x[2]),math.cos(math.pi/x[2]),x[0],x[1])
    i3 = i
    i4 = ndimage.filters.gaussian_filter(i3, 3)
    i5 = ndimage.filters.gaussian_filter(i2, 3)

    
    
    dyObjetivo, dxObjetivo = np.gradient(i4)
    # dzObjetivo = dxObjetivo + dyObjetivo
    
    dy, dx = np.gradient(i5)
    # dz = dx + dy
    
    distObjetivo = ndimage.morphology.distance_transform_edt(i4)
    
    distOriginal = ndimage.morphology.distance_transform_edt(i5)

    return np.sum(( np.power([dyObjetivo - dy],2)))+ np.sum(( np.power([dxObjetivo - dx],2))) + np.sum(( np.power([distObjetivo - distOriginal],2)))*100 + np.sum(( np.power([i5 - i4],2)))*0.5
    


def affin(i,a1=1,a2=0,a3=0,a4=1,t1=0,t2=0):            
    i2 = scipy.ndimage.interpolation.affine_transform(i,[[a1,a2],[a3,a4]],offset=[t1,t2],cval=0.0)
    return i2

def compare(image1, image2):
  return np.sum((image1 - image2) ** 2)

def register_point(image1, image2, point1, search_size = 100):
  print(point1)
  w, h = point1
  half = search_size // 2
  point2 = 0, 0
  error2 = np.inf
  i4 = ndimage.filters.gaussian_filter(image2, 3)
  i5 = ndimage.filters.gaussian_filter(image1, 3)

  edgesObjetivo = feature.canny(image2,sigma=1)
  edges = feature.canny(image1,sigma=1)


  distOriginal = ndimage.morphology.distance_transform_edt(np.logical_not(edges))
  distObjetivo = ndimage.morphology.distance_transform_edt(np.logical_not(edgesObjetivo))


    
  dyObjetivo, dxObjetivo = np.gradient(i4)
    # dzObjetivo = dxObjetivo + dyObjetivo
    
  dy, dx = np.gradient(i5)
    # dz = dx + dy
    
  distObjetivo = ndimage.morphology.distance_transform_edt(i4)
    
  distOriginal = ndimage.morphology.distance_transform_edt(i5)

  errorall = []
  for i in range(w - half, w + half):
    for j in range(h - half, h + half):
      pointi = (i, j)
      thumb1 = utils.thumb(image1, point1)
      thumbi = utils.thumb(image2, pointi)
      thumbDx = utils.thumb(dx,point1)
      thumbDxObjetivo = utils.thumb(dxObjetivo, pointi)
      thumbDy = utils.thumb(dy,point1)
      thumbDyObjetivo = utils.thumb(dyObjetivo, pointi)
      thumbDistOriginal = utils.thumb(distOriginal,point1)
      thumbDistObjetivo = utils.thumb(distObjetivo, pointi)


      #errori = E(0,thumb1, thumbi)
      errori = np.sum(( np.power([thumbDyObjetivo - thumbDy],2)))+ np.sum(( np.power([thumbDxObjetivo - thumbDx],2))) + np.sum(( np.power([thumbDistObjetivo - thumbDistOriginal],2)))*200 +  np.sum(( np.power([thumbi - thumb1],2)))*5

      # np.sum(( np.power([thumbDistObjetivo - thumbDistOriginal],2)))*100 +
      errorall.append(int(errori))
      if errori <= error2:
        point2 = pointi
        error2 = errori
  # print (errorall)
  return point2

def register_points(image1, image2, points):
  return [register_point(image1, image2, pointi) for pointi in points]

def register(images):
  points = [utils.ask_points(images[0])]
  for i in range(1, len(images)):
    image1 = utils.read(images[i - 1])
    image2 = utils.read(images[i])
    point1 = points[i - 1]
    point2 = register_points(image1, image2, point1)
    points.append(point2)
  print (points)
  return points

if __name__ == '__main__':
  path = sys.argv[1]
  images = utils.images(path)
  points = register(images)
  utils.render_points(images, points)
