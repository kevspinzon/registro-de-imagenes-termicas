import cv2

class Cli:
  def __init__(self, image, wsize = 30, hsize =30):
    self.image = image.copy()
    self.wsize = wsize
    self.hsize = hsize
    self.points = []

  def handle_click(self, event, y, x, flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
      # half = self.wsize // 2
     
      uHalf = self.wsize // 2
      lHalf = self.hsize // 2
      
      upper = y - uHalf, x - lHalf
      lower = y + uHalf, x + lHalf
      self.points.append((x, y))
      cv2.rectangle(self.image, upper, lower, 255, 1)

  def ask_points(self):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', self.handle_click)
    while cv2.waitKey(1) != 27:
      cv2.imshow('image', self.image)
    cv2.destroyAllWindows()
    return self.points

if __name__ == '__main__':
  image = cv2.imread('./images/patient4/1_0.png', 0)
  cli = Cli(image)
  print(cli.ask_points())
