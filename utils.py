import re
import os
import sys
import cv2
from cli import Cli

def read(path):
  clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
  image = cv2.imread(path, 0)
  blurred = cv2.blur(image, (3, 3))
  return clahe.apply(blurred)

def ask_points(image):
  cli = Cli(read(image))
  return cli.ask_points()

def rectangle(image, point, thumb_size = 30):
  w, h = point
  image = image.copy()
  half = thumb_size // 2
  upper = h - half, w - half
  lower = h + half, w + half
  return cv2.rectangle(image, upper, lower, 255, 2)

def thumb(image, point, thumb_size = 30):
  w, h = point
  half = thumb_size // 2
  return image[w - half : w + half, h - half: h + half].copy()

def images(path):
  pattern = '([0-9|a-z]+)'
  images = os.listdir(path)
  images = [i for i in images if i.endswith('png')]
  images = sorted(images, key = lambda i: int(re.split(pattern, i)[3]))
  images = [os.path.join(path, i) for i in images]
  return images

def render_points(images, points):
  folder="./result/"+ images[0].split('/')[-2]

  if not os.path.exists(folder):
    os.makedirs(folder)
  for image, points in zip(images, points):
    result = read(image)
    for point in points:
      result = rectangle(result, point)

    cv2.imwrite(os.path.join(folder, image.split('/')[-1]), result)
