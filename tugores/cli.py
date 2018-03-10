import cv2

class Cli:
  def __init__(self, image, wsize = 30):
    self.image = image.copy()
    self.wsize = wsize
    self.points = []

  def handle_click(self, event, y, x, flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
      half = self.wsize // 2
      upper = y - half, x - half
      lower = y + half, x + half
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
  image = cv2.imread('./images/3_2.png', 0)
  cli = Cli(image)
  print(cli.ask_points())
