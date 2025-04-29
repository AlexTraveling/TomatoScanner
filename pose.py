from ultralytics import YOLO
import cv2
import numpy as np
import math
import time

from detect import detect


def get_distance(x, y):

   return ((x - 0.5)**2 + (y - 0.5)**2)**0.5


def extract(result):

   boxes = result.boxes
   category = boxes.cls
   xywhn = boxes.xywhn

   # fruit
   distance_min = 1.0
   record = -1
   for i in range(len(category)):
      if category[i] == 0:
         distance = get_distance(xywhn[i][0], xywhn[i][1])
         if distance < distance_min:
            distance_min = distance
            record = i
   point_fruit = xywhn[record]

   # carpopodium
   distance_min = 1.0
   only_up_half = 0.5
   record = -1
   for i in range(len(category)):
      if category[i] == 1 and xywhn[i][1] < only_up_half:
         distance = get_distance(xywhn[i][0], xywhn[i][1])
         if distance < distance_min:
            distance_min = distance
            record = i
   point_carpopodium = xywhn[record]

   return point_fruit, point_carpopodium


def get_angle(image_path, point_fruit, point_carpopodium):

   # Calculate the differences in x and y
   dx = point_fruit[0] - point_carpopodium[0]
   dy = point_fruit[1] - point_carpopodium[1]

   print(f'dx: {dx}')
   print(f'dy: {dy}')

   # Calculate the angle in radians
   angle_radians = math.atan2(dx, dy)

   # Convert the angle to degrees
   angle_degrees = math.degrees(angle_radians)

   # if dx * dy < 0:
   #    angle_degrees -= 180.0

   return angle_degrees


def get_angle_display(image_path, point_fruit, point_carpopodium):

   image = cv2.imread(image_path)

   # Get the dimensions
   height, width, _ = image.shape

   # Denormalize the coordinates to get pixel values
   points = []
   for box in [point_fruit, point_carpopodium]:
      x_center = int(box[0] * width)
      y_center = int(box[1] * height)
      points.append((x_center, y_center))

   # Extract the two points
   point1, point2 = points

   # Calculate the slope (m) and y-intercept (b) of the line
   if point2[0] != point1[0]:  # Avoid division by zero for vertical lines
      m = (point2[1] - point1[1]) / (point2[0] - point1[0])  # Slope
      b = point1[1] - m * point1[0]  # Intercept

      # Calculate the intersection points with the top and bottom of the image
      x_top = int(-b / m)  # x when y = 0 (top of image)
      x_bottom = int((height - b) / m)  # x when y = height (bottom of image)

      # Ensure the x-coordinates are within the image bounds
      x_top = max(0, min(width, x_top))
      x_bottom = max(0, min(width, x_bottom))
   else:  # Vertical line case
      x_top = point1[0]
      x_bottom = point1[0]

   # Draw the line across the full height
   cv2.line(image, (x_top, 0), (x_bottom, height), (0, 0, 255), thickness=2)  # Red line

   # Optionally draw the centers as circles for clarity
   for point in points:
      cv2.circle(image, point, 5, (255, 0, 0), -1)  # Blue dots

   return image


def rotate(image_path, angle):

   # Load the image
   image = cv2.imread(image_path)
   height, width, _ = image.shape

   # Compute the center of the image
   center = (width // 2, height // 2)

   # Get the rotation matrix
   rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

   # Rotate the image
   rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(0, 0, 0))

   return rotated_image


def pose(plant_id, region_id_list, conf):

   for id in region_id_list:

      print(f'--- {id} ---')

      image_name = f'region{id}'
      image_path = f'after_split/plant{plant_id}_{image_name}.jpg'

      print('【Predict】')
      result = detect(image_path, conf)
      after_predict_path = f'after_pose/plant{plant_id}_{image_name}_after1_predict.jpg'
      result.save(after_predict_path)

      print('【Extract】')
      point_fruit, point_carpopodium = extract(result)
      print(f'fruit:       {point_fruit}')
      print(f'carpopodium: {point_carpopodium}')

      print('【Angle】')
      angle = get_angle(image_path, point_fruit, point_carpopodium)
      angle_display = get_angle_display(after_predict_path, point_fruit, point_carpopodium)
      cv2.imwrite(f'after_pose/plant{plant_id}_{image_name}_after2_angle.jpg', angle_display)
      print(angle)

      print('【Rotate】')
      rotated_image = rotate(image_path, angle)
      cv2.imwrite(f'after_pose/plant{plant_id}_{image_name}_after3_rotate.jpg', rotated_image)

      time.sleep(1)


if __name__ == '__main__':

   pose(3, [2], conf=0.2)