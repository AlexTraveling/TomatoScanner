from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Point, Polygon
import math

from operate_database import write


def get_distance(x1, y1, x2, y2):

   return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def display_seg(xy, size):

   # Create a black background
   background = np.zeros((size, size, 3), dtype=np.uint8)

   # Colors for each object
   colors = [
      (0, 0, 255),  # Red for the first object
      (0, 255, 0),   # Green for the second object
      (255, 0, 0)
   ]

   # Plot points on the black background
   for i, points in enumerate(xy):
      color = colors[i % len(colors)]  # Cycle through colors
      for point in points:
         x, y = int(point[0]), int(point[1])
         if 0 <= x < size and 0 <= y < size:  # Ensure points are within the frame
               background[y, x] = color  # Draw the point

   return background


def get_center(xy, size):

   center_point = Point(size / 2, size / 2)
   
   for points in xy:
      polygon = Polygon(points)
      if polygon.contains(center_point):
         return points  


def delete_corner(xy, delete_rate):

   print('Delete corner.')
   print(f'original type: {type(xy)}')
   # print(xy)

   top = xy[:, 1].min()  # Minimum y-value (topmost point)
   bottom = xy[:, 1].max()  # Maximum y-value (bottommost point)
   left = xy[:, 0].min()  # Minimum x-value (leftmost point)
   right = xy[:, 0].max()  # Maximum x-value (rightmost point)

   width = right - left
   height = bottom - top

   if width > height:
      radius = width / 2
   else:
      radius = height / 2

   # print(f'radius: {radius}')
   
   center = (width / 2 + left, height / 2 + top)

   out = []

   for point in xy:

      distance = get_distance(point[0], point[1], center[0], center[1])
      # print(distance)
      if distance < radius * delete_rate:
         # out.append([int(point[0]), int(point[1])])
         out.append(point)

   out = np.array(out)

   return out


def get_width_height(center_xy):

   # Convert center_xy to numpy array if it is a list
   center_xy = np.array(center_xy)
   
   # Get the top, bottom, left, right points
   top = center_xy[:, 1].min()  # Minimum y-value (topmost point)
   bottom = center_xy[:, 1].max()  # Maximum y-value (bottommost point)
   left = center_xy[:, 0].min()  # Minimum x-value (leftmost point)
   right = center_xy[:, 0].max()  # Maximum x-value (rightmost point)

   print(top)
   print(bottom)
   print(left)
   print(right)

   height = bottom - top
   width = right - left

   top = int(top)
   bottom = int(bottom)
   left = int(left)
   right = int(right)
   
   return height, width, top, bottom, left, right


def display_width_height(after_clean, top, bottom, left, right):
   # Make a copy of the original mask image to draw on
   image_with_lines = after_clean.copy()

   # Set line color (turquoise) and thickness
   line_color = (224, 64, 208)  # RGB for turquoise
   line_color = (255, 255, 255)
   line_thickness = 2

   # Draw horizontal lines
   cv2.line(image_with_lines, (left, top), (right, top), line_color, line_thickness)  # Top line
   cv2.line(image_with_lines, (left, bottom), (right, bottom), line_color, line_thickness)  # Bottom line

   # Draw vertical lines
   cv2.line(image_with_lines, (left, top), (left, bottom), line_color, line_thickness)  # Left line
   cv2.line(image_with_lines, (right, top), (right, bottom), line_color, line_thickness)  # Right line

   font_size = 1.2

   # Width text (above the top line)
   width_text = f"Width: {right - left} px"
   (text_width, _), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   cv2.putText(image_with_lines, width_text, 
               (int((left + right) / 2) - int(text_width / 2), top - 10),  # Centered above the top line
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)

   # Height text (to the left of the left line, vertically rotated)
   height_text = f"{bottom - top} px"
   # rotation_matrix = cv2.getRotationMatrix2D((0, 0), 90, 1) 
   (text_width, text_height), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   cv2.putText(image_with_lines, height_text, 
               (int(right) + 10, int((top + bottom) / 2)), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)

   return image_with_lines


def get_area(xy):

   # Convert points into numpy array for easy calculation
   xy = np.array(xy, dtype=np.int32)
   hull = cv2.convexHull(xy)
   area = cv2.contourArea(hull)

   return area, hull


def display_area(area, hull, after_width_height, size, bottom, if_polygon):
   
   after_area = after_width_height.copy()

   # for point in xy:
   #    cv2.circle(after_area, tuple(point), 2, (0, 0, 255), -1)  # Red points

   # Draw the convex hull polygon
   if if_polygon:
      cv2.polylines(after_area, [hull], isClosed=True, color=(0, 255, 0), thickness=1)  # Green polygon
   
   # Display the area on the image
   font_size = 1.2
   # area_text = f"Area: {area:.2f} px2"
   area_text = f"Area: {int(area)} px2"
   (text_width, text_height), text_bottom = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   # text_position = (int(size / 2 - text_width / 2), size - text_height - text_bottom - 10)
   text_position = (int(size / 2 - text_width / 2), bottom + 50 + text_height + text_bottom)
   cv2.putText(after_area, area_text, text_position, 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
   
   return after_area


def get_diameter(image, size, top, bottom):

   diameters = []

   # Iterate through all y-values in the image
   for y in range(top, bottom + 1):
      # Extract all x-values (green pixels) at the current y-value
      green_pixels = []
      for x in range(size):
         # Check for the green color (0, 255, 0)
         if np.array_equal(image[y, x], [0, 255, 0]):  # Green pixel
            green_pixels.append(x)

      # Handle the special cases
      if len(green_pixels) == 0:
         # No green pixels at this y value, set diameter to 0
         diameters.append(0)
      elif len(green_pixels) == 1:
         # Only one green pixel, set diameter to 0 (likely at the top or bottom)
         diameters.append(0)
      else:
         # More than one green pixel, keep the leftmost and rightmost
         left = min(green_pixels)
         right = max(green_pixels)
         diameter = right - left
         diameters.append(diameter)

   diameter_list = diameters
   return diameter_list


def get_volume(diameter_list):

   volume = 0.0

   for diameter in diameter_list:
      radius = float(diameter) / 2
      volume += math.pi * (radius ** 2)

   return volume


def display_volume(volume, after_area, size, bottom):
   
   after_volume = after_area.copy()

   # Display the area on the image
   font_size = 1.2
   # volume_text = f"Volume: {volume:.2f}px3"
   volume_text = f"Volume: {int(volume)} px3"
   # (text_width, _), _ = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   # text_position = (int(size / 2 - text_width / 2), size - 10)
   (text_width, text_height), text_bottom = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   text_position = (int(size / 2 - text_width / 2), bottom + 50 + text_height + text_bottom + 10 + text_height + text_bottom)
   cv2.putText(after_volume, volume_text, text_position, 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
   
   return after_volume


def make_mask(image_path, points_array):

   image = cv2.imread(image_path)

   # Convert the points array to integer type
   points = points_array.astype(int)

   # Create a mask with the same size as the image
   mask = np.zeros_like(image, dtype=np.uint8)

   # Fill the polygon with blue color (BGR format: (255, 0, 0))
   cv2.fillPoly(mask, [points], color=(255, 0, 0))

   # Blend the mask with the original image (alpha = 0.5 for transparency)
   blended = cv2.addWeighted(image, 1.0, mask, 0.5, 0)

   return blended


def seg(plant_id, region_id, imgsz=640):

   # Segmentation
   # image_path = "input-2.jpg"
   # id = 3
   region_name = f'region{region_id}'
   image_path = f"after_EdgeBoost/plant{plant_id}_{region_name}_after3_rotate_EdgeBoost.jpg"
   image = Image.open(image_path)
   width, height = image.size
   if width == height:
      size = width

   EdgeYOLO = YOLO('weights/EdgeYOLO.pt')
   results = EdgeYOLO.predict(image_path,
                  save=True,
                  conf=0.5, 
                  iou=0.5,
                  imgsz=imgsz,
                  save_txt=True
                  #  show_boxes=False
                  )

   result = results[0]
   result.save(f'after_seg/plant{plant_id}_{region_name}_after4_seg.png')
   masks = result.masks
   data = masks.data
   data = data[0]
   xy = masks.xy

   after_seg = display_seg(xy, size)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after5_edge.png", after_seg)

   # Clean
   center_xy = get_center(xy, size)
   # delete corner
   center_xy = delete_corner(center_xy, 1.1)
   after_clean = display_seg([center_xy], size)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after6_clean.png", after_clean)
   # save clean
   np.save(f'after_seg/plant{plant_id}_region{region_id}_seg_result.npy', np.array(center_xy))

   # Get width and height
   height, width, top, bottom, left, right = get_width_height(center_xy)
   print(f'width: {width}')
   print(f'height: {height}')
   after_width_height = display_width_height(after_clean, top, bottom, left, right)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after7_width_height.png", after_width_height)

   # Get area
   area, hull = get_area(center_xy)
   print(f'area: {area}')
   after_area = display_area(area, hull, after_width_height, size, bottom, True)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after8_area.png", after_area)

   # Get volume
   diameter_list = get_diameter(cv2.imread(f'after_seg/plant{plant_id}_{region_name}_after8_area.png'), size, top, bottom)
   # print(f'diameter: {diameter_list}')
   volume = get_volume(diameter_list)
   print(f'vulume: {volume}')
   after_volume = display_volume(volume, after_area, size, bottom)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after9_volume.png", after_volume)

   write(f'{plant_id}-{region_id}', 'Wpixel', width)
   write(f'{plant_id}-{region_id}', 'Hpixel', height)
   write(f'{plant_id}-{region_id}', 'Apixel', area)
   write(f'{plant_id}-{region_id}', 'Vpixel', volume)

   # Make mask

   # # Step 2
   # center_xy = np.load('center_xy.npy')

   after_mask = make_mask(image_path, center_xy)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after10_mask.png", after_mask)
   print('mask success')

   after_mask = display_width_height(after_mask, top, bottom, left, right)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after11_mask_width_height.png", after_mask)

   after_mask = display_area(area, hull, after_mask, size, bottom, False)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after12_mask_width_height_area.png", after_mask)
   
   after_mask = display_volume(volume, after_mask, size, bottom)
   cv2.imwrite(f"after_seg/plant{plant_id}_{region_name}_after13_mask_width_height_area_volume.png", after_mask)
   
   
if __name__ == '__main__':
   
   seg(0, 2, 960)