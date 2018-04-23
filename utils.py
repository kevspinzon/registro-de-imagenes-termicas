import re
import os
import sys
import cv2
from cli import Cli
import register
from optparse import OptionParser

def read(path):
  clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
  image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  blurred = cv2.blur(image, (3, 3))
  return clahe.apply(blurred)

def ask_points(image):
  register.init()
  cli = Cli(read(image),register.ops.horizontalLenght,register.ops.verticalLenght)
  return cli.ask_points()

def rectangle(image, point, thumb_size = 30):
  w, h = point
  image = image.copy()
  # half = thumb_size // 2
  wHalf=register.ops.verticalLenght // 2
  hHalf=register.ops.horizontalLenght // 2
  upper = h - hHalf, w - wHalf
  lower = h + hHalf, w + wHalf
  return cv2.rectangle(image, upper, lower, 255, 2)

def thumb(image, point, thumb_size = 30):
  w, h = point
  # half = thumb_size // 2
  uHalf = register.ops.horizontalLenght // 2
  lHalf = register.ops.verticalLenght // 2
  return image[w - lHalf : w + lHalf, h - uHalf: h + uHalf].copy()

def images(path):
  pattern = '([0-9|a-z]+)'
  images = os.listdir(path)
  images = [i for i in images if i.endswith('png')]
  images = sorted(images, key = lambda i: int(re.split(pattern, i)[3]))
  images = [os.path.join(path, i) for i in images]
  return images

def render_points(images, points,path):
  # folder="./result/"+ images[0].split('/')[-2]

  if not os.path.exists(path):
    os.makedirs(path)
  for image, points in zip(images, points):
    result = read(image)
    for point in points:
      result = rectangle(result, point)

    cv2.imwrite(os.path.join(path, image.split('/')[-1]), result)

def optParse():
    parser = OptionParser(usage="usage: %prog [options] $image_folder",
                          version="%prog 2.0")
    
    parser.add_option("-d", "--distance",
                      action="store", # optional because action defaults to "store"
                      dest="weightDistance",
                      default= 1.35,
                      type="float",
                      help="Peso de la transformación distancia")

    parser.add_option("-g", "--gradient",
                      action="store", # optional because action defaults to "store"
                      dest="weightGradient",
                      default=1,
                      type="float",
                      help="Peso de la transformación gradiente",)

    parser.add_option("-p", "--pixel",
                      action="store", # optional because action defaults to "store"
                      dest="weightPixel",
                      default=1,
                      type="float",
                      help="Peso del error por pixel",)
     
    parser.add_option("-i", "--inputPoints",
                      action="store", # optional because action defaults to "store"
                      dest="inputPoints",
                      default=False,
                      help="Archivo con los puntos de entrada",)

    parser.add_option("-e", "--Exit_Folder",
                      action="store", # optional because action defaults to "store"
                      dest="exitFolder",
                      default='./result/',
                      help="Folder de salida",)

    parser.add_option('--hard', 
                        dest='operation', 
                        action='store_const',
                        const='hardmode', 
                        default='optimize',
                        help='Deternmina si se utiliza la busqueda intensiva. Por defecto utiliza  Bashhoping de  Scipy ')

    parser.add_option("--vl",
                      action="store", # optional because action defaults to "store"
                      dest="verticalLenght",
                      default=50,
                      type="int",
                      help="Longitud vertical de la ventana. Default = 30")

    parser.add_option("--hl",
                      action="store", # optional because action defaults to "store"
                      dest="horizontalLenght",
                      default=50,
                      type="int",
                      help="Longitud vertical de la ventana. Default =30",)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("wrong number of arguments")

    return options,args
