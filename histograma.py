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


def affin(i,a1=1,a2=0,a3=0,a4=1,t1=0,t2=0):            
    i2 = scipy.ndimage.interpolation.affine_transform(i,[[a1,a2],[a3,a4]],offset=[t1,t2],cval=0.0)
    return i2

def getTransformaciones(image):
    transformations = {'original': [] , 'dx' :  [] , 'dy': [], 'distancia' :  [] }
    
    transformations['original'] = ndimage.filters.gaussian_filter(image, 2)
    transformations['distancia'] = np.array(ndimage.morphology.distance_transform_edt(np.logical_not(feature.canny(image,sigma=0))), dtype=np.uint32)
    transformations['dy'] , transformations['dx']  = np.gradient(image)

    return transformations

def getTransformedThumbs(transformations=[],position=[],):
  thumbs = {'original': [] , 'dx' :  [] , 'dy': [], 'distancia' :  [] }

  thumbs['original'] = utils.thumb(transformations['original'], position)
  thumbs['dx'] = utils.thumb(transformations['dx'], position)
  thumbs['dy'] = utils.thumb(transformations['dy'], position)
  thumbs['distancia'] = utils.thumb(transformations['distancia'] ,position)

  return thumbs




def register_point(pointOri,imageOriginal, imageObjetivo):  
  
  originalThumbs= getTransformedThumbs(getTransformaciones(imageOriginal),pointOri)
  objectiveTransformations = getTransformaciones(imageObjetivo);


  if (ops.operation=='optimize'):

    result = scipy.optimize.basinhopping(calculateError, x0 = pointOri , stepsize=1, minimizer_kwargs={'args':(originalThumbs, objectiveTransformations), 'method':'Nelder-Mead'})
    
    pointObj= [int(result.x[0]),int(result.x[1])]

  else:

    half = 50 // 2
    rranges = (slice(pointOri[0] - half, pointOri[0] + half, 1), slice(pointOri[1] - half, pointOri[1] + half, 1))

    result = scipy.optimize.brute(calculateError, rranges , args=(originalThumbs, objectiveTransformations))
    
    pointObj= [int(result[0]),int(result[1])]
 
  return pointObj

def calculateError(current,originalThumbs,transformations):
  
  objectiveThumbs= getTransformedThumbs(transformations, [int(current[0]), int(current[1])] )
      

  try:
    errorIntensidades  = np.sum(( np.power([originalThumbs['original'] - objectiveThumbs['original']],2))) * ops.weightPixel

    errorGradienteY   =  np.sum(( np.power([originalThumbs['dy'] - objectiveThumbs['dy']],2)))   

    errorGradienteX   =  np.sum(( np.power([originalThumbs['dx'] - objectiveThumbs['dx']],2)))

    errorGradiente    =  (errorGradienteX + errorGradienteY) * ops.weightGradient
    
    errorDistancia    = np.sum((np.power([originalThumbs['distancia'] - objectiveThumbs['distancia']],2)))* ops.weightDistance

    errorTotal= errorIntensidades + errorDistancia + errorGradiente
  except: 
    errorTotal= np.inf

  return errorTotal


def register_points(image1, image2, points):
 return [register_point(pointi,image1, image2) for pointi in points]



def CDF(histograma):
    K = histograma.size
    n = np.sum(histograma[0:K-1])

    P = np.zeros(K)
    c = histograma[0]
    P[0]=c/n

    for i in range (1,K):
        c = c+histograma[i]
        P[i] = c/n
    return P


def register(images):
  if (ops.inputPoints != False):
     file = np.genfromtxt(ops.inputPoints)
     data = [[( int(file[x][0]), int(file[x][1]))  for x in range(len(file))]]
  else:
    data = [utils.ask_points(images[0])]

  points = data

  t_images = []

  print (points)

  lengImages=len(images)
  theta = 350
  
  for i in range(1, lengImages):
    print("+Imagen:",i," de: ", lengImages-1)

    if (i == 1):
      image1 = utils.read(images[i - 1])
    else:
      image1 = np.copy(image2)
    if (i%2 == 0):
      image2 = affin(image1,np.cos(np.pi/theta),np.sin(np.pi/theta),-np.sin(np.pi/theta),np.cos(np.pi/theta),2,2)
    else:
      image2 = affin(image1,1,0,0,1,2,2)
    t_images.append(image2)
    ha,bin_edgesa = np.histogram(image1,256)
    hr,bin_edgesr = np.histogram(image2,256)

    K = ha.size
    Pa = CDF(ha)
    Pr = CDF(hr)
    fhs = np.zeros(K)
    for a in range(K):
      j=K-1
      while True:
        fhs[a] = j
        j=j-1
        if (j < 0 or Pa[a] > Pr[j]):
          break
          
          
    h,w = image1.shape
    for ip in range(h):
      for jp in range(w):
        inda=np.searchsorted(bin_edgesa,image1[ip][jp])
        indr=fhs[inda-1]
        image1[ip][jp] = bin_edgesr[int(indr-1)]

    point1 = points[i - 1]
    point2 = register_points(image1, image2, point1)
    points.append(point2)

  return points,t_images

def init():
  global ops
  global args
  ops,args = utils.optParse()


if __name__ == '__main__':
  init()
  # print (args,ops)
  path = args[0]
  images = utils.images(path)
  points,t_images = register(images)
  print (images)
  print (t_images)
  utils.render_pointsHist(images,t_images, points,ops.exitFolder)
