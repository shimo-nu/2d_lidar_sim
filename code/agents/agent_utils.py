import numpy as np
class LocalGoal:
  def __init__(self, position, range, area = None):
    self.x = position[0]
    self.y = position[1]
    self.area = area
    self.range = range
    self.thres = 0.2
    self.area_value = 0
    
  def isClear(self,area = None, use_range = False):
    if (area is None):
      area = self.area
    elif (use_range):
      # print(area)
      # print(area.shape)
      y_range = [int(max(0, self.y - self.range)), int(max(area.shape[1], self.y + self.range))]
      x_range = [int(max(0, self.x - self.range)), int(max(area.shape[0], self.x + self.range))]
      # print(y_range)
      # print(x_range)
      area = area[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    data_size = area.shape
    self.area_value = np.sum(area) / (data_size[0] * data_size[1])
    if (self.area_value < self.thres):
      return True
    else:
      return False
    
  def __array__(self):
        # ここで必要な属性をNumPy配列に変換
        return np.array([self.x, self.y])
      
  def __getitem__(self, index):
      if index == 0:
          return self.x
      elif index == 1:
          return self.y
      else:
          raise IndexError("Index out of range")


  def __str__(self):
      return f"LocalGoal(x={self.x}, y={self.y}, v={self.area_value})"
    
  def __repr__(self):
      return f"LocalGoal(x={self.x}, y={self.y}, v={self.area_value})"