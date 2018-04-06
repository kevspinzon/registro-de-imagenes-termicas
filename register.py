from scipy import misc
from scipy import ndimage
from scipy import optimize
from skimage import feature

import matplotlib.pyplot as plt


import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils
import scipy
from numba import vectorize, float64


#x= parametros de transformacion i=imagen original i2=imagen objetivo
#@vectorize(['int64(int64, int64)'], target='cpu')
def E(i,i2):
    

    
    
    dyObjetivo, dxObjetivo = np.gradient(i)
    # dzObjetivo = dxObjetivo + dyObjetivo
    
    dy, dx = np.gradient(i2)
    # dz = dx + dy
    
    #distObjetivo = ndimage.morphology.distance_transform_edt(i4)
    
    #distOriginal = ndimage.morphology.distance_transform_edt(i5)

    #return (np.sum(( np.power([dyObjetivo - dy],2)))*2 + np.sum(( np.power([dxObjetivo - dx],2))))*2 + np.sum(( np.power([thumb1 - thu],2)))*0.75
    


def affin(i,a1=1,a2=0,a3=0,a4=1,t1=0,t2=0):            
    i2 = scipy.ndimage.interpolation.affine_transform(i,[[a1,a2],[a3,a4]],offset=[t1,t2],cval=0.0)
    return i2

def compare(image1, image2):
  return np.sum((image1 - image2) ** 2)

def register_point(point1,image1, image2):
  # print(point1)
  # w, h = point1
  # half = search_size // 2
  # point2 = 0, 0
  # error2 = np.inf
  
  
  gaussianImage1 = ndimage.filters.gaussian_filter(image1, 2)
  gaussianImage2 = ndimage.filters.gaussian_filter(image2, 2)
  
  
  edges = feature.canny(gaussianImage2,sigma=0)
  edgesObjetivo = feature.canny(gaussianImage1,sigma=0)


  distOriginal = ndimage.morphology.distance_transform_edt(np.logical_not(edges))

  #distOriginal = ndimage.morphology.distance_transform_edt(np.logical_not(edges))
  dist16Orig = np.array(distOriginal, dtype=np.uint16) # This line only change the type, not values
  dist16Orig *= 256

  distObjetivo = ndimage.morphology.distance_transform_edt(np.logical_not(edgesObjetivo))
  dist16Objt = np.array(distObjetivo, dtype=np.uint16) # This line only change the type, not values
  dist16Objt *= 256


  dyObjetivo, dxObjetivo = np.gradient(image2)

    
  dy, dx = np.gradient(image1)
  

  if (ops.operation=='optimize'):
    result = scipy.optimize.basinhopping(calculateError, point1 , minimizer_kwargs={'args':(point1,image1,image2,dx,dxObjetivo,dy,dyObjetivo,dist16Orig,dist16Objt), 'method':'Nelder-Mead'})
    point2= [int(result.x[0]),int(result.x[1])]

  else:
    point2 = hardmodeSearch(point1,image1,image2,dx,dxObjetivo,dy,dyObjetivo,dist16Orig,dist16Objt)



  
  
  return point2

def  hardmodeSearch(point1,image1,image2,dx,dxObjetivo,dy,dyObjetivo,dist16Orig,dist16Objt,search_size=100):
  w, h = point1
  half = search_size // 2
  point2 = 0, 0
  error2 = np.inf
  for i in range(w - half, w + half):
    for j in range(h - half, h + half):
      pointi =[i,j]
      errori = calculateError(pointi,point1,image1,image2,dx,dxObjetivo,dy,dyObjetivo,dist16Orig,dist16Objt)

      # print (errori)

      if errori <= error2:
        point2 = pointi
        error2 = errori
  
  return point2

def calculateError(index,point1,image1,image2,dx,dxObjetivo,dy,dyObjetivo,dist16Orig,dist16Objt):
  


  pointi = (int(index[0]), int(index[1]))
      
  # print(point1,pointi)

  thumb1 = utils.thumb(image1, point1)
  thumbi = utils.thumb(image2, pointi)

  thumbDx = utils.thumb(dx,point1)
  thumbDxObjetivo = utils.thumb(dxObjetivo, pointi)
  
  thumbDy = utils.thumb(dy,point1)
  thumbDyObjetivo = utils.thumb(dyObjetivo, pointi)

  thumbdisOri = utils.thumb(dist16Orig ,pointi)
  thumbdisObj = utils.thumb(dist16Objt ,point1)



  try:
    errorIntensidades = + np.sum(( np.power([thumb1 - thumbi],2))) * ops.weightPixel      
    errorGradiente =  np.sum(( np.power([thumbDyObjetivo - thumbDy],2))) + np.sum(( np.power([thumbDxObjetivo - thumbDx],2))) * ops.weightGradient
    errorDistancia = np.sum((np.power([thumbdisObj - thumbdisOri],2)))* ops.weightDistance

    error1= errorIntensidades + errorDistancia + errorGradiente
  except: 
    error1= np.inf

  return error1


def register_points(image1, image2, points):
 return [register_point(pointi,image1, image2) for pointi in points]

def register(images):
  

  if (ops.inputPoints != False):
     file = np.genfromtxt(input)
     data = [[( int(file[x][0]), int(file[x][1]))  for x in range(len(file))]]
  else:
    data = [utils.ask_points(images[0])]

  points = data

  print (points)

  lengImages=len(images)

  for i in range(1, lengImages):
    print("+Imagen:",i," de: ", lengImages-1)
    

    image1 = utils.read(images[i - 1])
    image2 = utils.read(images[i])
    point1 = points[i - 1]
    point2 = register_points(image1, image2, point1)
    points.append(point2)
  #print (points)
  return points

def init():
  global ops
  global args
  ops,args = utils.optParse()


if __name__ == '__main__':
  init()
  print (args,ops)
  path = args[0]
  images = utils.images(path)
  points = register(images)
  utils.render_points(images, points,ops.exitFolder)
