
from scipy import misc
from scipy import ndimage
from scipy import optimize
from skimage import feature

import cv2
import scipy
import imageio
import math
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random

def affin(i,a1=1,a2=0,a3=0,a4=1,t1=0,t2=0):
    i2 = scipy.ndimage.interpolation.affine_transform(i,[[a1,a2],[a3,a4]],offset=[t1,t2],cval=0.0)
    return i2



def E(x,i,i2):
    #Solo para traslado en el eje X y Y
    #i3 = affin(i,1,0,0,1,x[0],x[1]) 

    # i3 = affin(i,math.cos(math.pi/x[2]),math.sin(math.pi/x[2]),-math.sin(math.pi/x[2]),math.cos(math.pi/x[2]),x[0],x[1])
    i4 = ndimage.filters.gaussian_filter(i, 3)
    i5 = ndimage.filters.gaussian_filter(i2, 3)

    #detector de bordes para la transofrmacion distancia
    edgesObjetivo = feature.canny(i,sigma=1)
    edges = feature.canny(i2,sigma=1)


    distOriginal = ndimage.morphology.distance_transform_edt(np.logical_not(edges))
    distObjetivo = ndimage.morphology.distance_transform_edt(np.logical_not(edgesObjetivo))
    distObjetivo = distObjetivo * 200
    distOriginal = distOriginal *200

    dyObjetivo, dxObjetivo = np.gradient(i4)
    
    dy, dx = np.gradient(i5)
    
    
    print ("Minimo Gradiente y:")
    print (np.amin(dyObjetivo))
    print ("Minimo Gradiente x:")
    print (np.amin(dxObjetivo))
    print ("Minimo Distancia:")
    print (np.amin(distObjetivo))
    print ("Minimo valor pixel:")
    print (np.amin(i4))
    print ("\n")
    
    print ("Maximo Gradiente y:")
    print (np.amax(dyObjetivo))
    print ("Maximo Gradiente x:")
    print (np.amax(dxObjetivo))
    print ("Maximo Distancia:")
    print (np.amax(distObjetivo))
    print ("Maximo valor pixel:")
    print (np.amax(i4))
    print ("\n")

    print ("sum gradiente y")
    print (np.sum(dyObjetivo))
    print ("sum gradiente x")
    print (np.sum(dxObjetivo))
    print ("sum Distancia")
    print (np.sum(distObjetivo))
    print ("sum pixel")
    print (np.sum(i4)*5)
    print ("\n")
    
    print("diferencia Gradiente Y")
    test = np.sum(( np.power([dyObjetivo - dy],2)))
    print (test)

    print("diferencia Gradiente X")
    test2 = np.sum(( np.power([dxObjetivo - dx],2)))
    print(test2)

    print("Diferencia de distrancias")
    test3 = np.sum(( np.power([distObjetivo - distOriginal],2)))
    print (test3)


    # test4 = np.sum(( np.power([dyObjetivo - dy],2))) * 0.015 + np.sum(( np.power([dxObjetivo - dx],2))) * 0.015 + np.sum(( np.power([distObjetivo - distOriginal],2)))
    print ("diferencia de intensidades")
    test5 = np.sum(( np.power([i5 - i4],2)))*5
    print (test5)
    print (x)
    
    return np.sum(( np.power([dyObjetivo - dy],2)))+ np.sum(( np.power([dxObjetivo - dx],2))) + np.sum(( np.power([distObjetivo - distOriginal],2)))*100 + np.sum(( np.power([i5 - i4],2)))*0.5
    



# i = scipy.misc.face(True)
i = imageio.imread("images/patient1/1_0.png")
# i2= affin(i,1,0,0,1,20,15)
#i2 = affin(i,math.cos(math.pi/75),math.sin(math.pi/75),-math.sin(math.pi/75),math.cos(math.pi/75),20,15)
i2 = imageio.imread("images/patient1/1_1.png")


print (E(0,i,i2))


# plt.subplot(121)
# plt.imshow(i,cmap='gray')
# plt.axis('off')
# plt.title('$I_1$')

# plt.subplot(122)
# plt.imshow(i2,cmap='gray')
# plt.axis('off')
# plt.title('$I_2$')


# plt.show()




i3 = ndimage.filters.gaussian_filter(i, 4)

i4 = feature.canny(i,sigma=1)
dist = ndimage.morphology.distance_transform_edt(np.logical_not(i4))
# print (dist[0])


plt.subplot(111)
plt.imshow(dist,cmap='gray')
plt.axis('off')
plt.title('$I_2$')


plt.show()

# print (E([10,15,75],i,i2))


# # In[6]:



# print (E([20,15,75],i,i2))


# # Registro simple con fmin

# In[7]:


#Traslado de 100 Pixeles a la izquierda
#i2 = affin(i,math.cos(math.pi/16),math.sin(math.pi/16),-math.sin(math.pi/16),math.cos(math.pi/16),20,15)
# i2 = imageio.imread("1_1.png")
# plt.imshow(i2,cmap='gray')
# plt.title('I')


# # In[8]:


# op = optimize.minimize(E,[8,8,20],args=(i,i2), method='Nelder-Mead')
# print (op)


# # In[9]:


# print (op.x)


# # In[10]:


# afin = affin(i,math.cos(math.pi/op.x[2]),math.sin(math.pi/op.x[2]),-math.sin(math.pi/op.x[2]),math.cos(math.pi/op.x[2]),op.x[0],op.x[1])
# plt.imshow(afin,cmap='gray')
# plt.title('I')


# # In[11]:


# xxx = i - afin
# plt.imshow(xxx,cmap='gray')
# plt.title('I')


# # In[6]:


# prueba = affin(i,1,0,0,1,5,15)
# op = optimize.minimize(E,[10,10],args=(i,prueba), method='Nelder-Mead')


# # In[7]:


# print (op)


# # In[20]:


# recuperada = affin(i,1,0,0,1,op.x[0],op.x[1])

# resta = prueba - recuperada

# plt.imshow(resta,cmap='gray')
# plt.title('I')
# print (np.mean(np.abs(resta)))

