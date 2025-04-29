import torch
import cv2
import numpy as np
from PIL import Image

from operate_database import read


def make_mask(image_path, points_array):

   image = cv2.imread(image_path)

   # Convert the points array to integer type
   points = points_array.astype(int)

   # Create a mask with the same size as the image
   mask = np.zeros_like(image, dtype=np.uint8)

   # Fill the polygon with blue color (BGR format: (255, 0, 0))

   fill_color = (255, 255, 255)
   edge_color = (255, 255, 255)
   # fill_color = (00, 33, 169)
   # edge_color = (00, 33, 169)
   # fill_color = (255, 00, 00)
   # edge_color = (255, 00, 00)
   fill_alpha = 0.3
   edge_alpha = 0.7
   thickness = 3

   cv2.fillPoly(mask, [points], color=fill_color)
   # Blend the mask with the original image (alpha = 0.5 for transparency)
   blended = cv2.addWeighted(image, 1, mask, fill_alpha, 0)

   mask = np.zeros_like(image, dtype=np.uint8)
   cv2.polylines(mask, [points], isClosed=True, color=edge_color, thickness=thickness)
   blended = cv2.addWeighted(blended, 1, mask, edge_alpha, 0)

   return blended


def display_width_height(input, top, bottom, left, right, w, h):
   
   # Make a copy of the original mask image to draw on
   image_with_lines = input.copy()

   # Set line color (turquoise) and thickness
   line_color = (224, 64, 208)  # RGB for turquoise
   line_color = (255, 255, 255)
   line_thickness = 3

   top, bottom, left, right = int(top), int(bottom), int(left), int(right)

   # Draw horizontal lines
   cv2.line(image_with_lines, (left, top), (right, top), line_color, line_thickness)  # Top line
   cv2.line(image_with_lines, (left, bottom), (right, bottom), line_color, line_thickness)  # Bottom line

   # Draw vertical lines
   cv2.line(image_with_lines, (left, top), (left, bottom), line_color, line_thickness)  # Left line
   cv2.line(image_with_lines, (right, top), (right, bottom), line_color, line_thickness)  # Right line

   font_size = 1.2

   # Width text (above the top line)
   width_text = f"{w:.3f} cm"
   (text_width, _), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   cv2.putText(image_with_lines, width_text, 
               (int((left + right) / 2) - int(text_width / 2), top - 10),  # Centered above the top line
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)

   # Height text (to the left of the left line, vertically rotated)
   height_text = f"{h:.3f} cm"
   # rotation_matrix = cv2.getRotationMatrix2D((0, 0), 90, 1) 
   (text_width, text_height), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   cv2.putText(image_with_lines, height_text, 
               (int(right) + 10, int((top + bottom) / 2)), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)

   return image_with_lines


def display_area(input, a, size, bottom):

   bottom = int(bottom)
   
   after_area = input.copy()

   # Display the area on the image
   font_size = 1.2
   # area_text = f"Area: {area:.2f} px2"
   area_text = f"Area: {a:.3f} cm2"
   (text_width, text_height), text_bottom = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   # text_position = (int(size / 2 - text_width / 2), size - text_height - text_bottom - 10)
   text_position = (int(size / 2 - text_width / 2), bottom + 50 + text_height + text_bottom)
   cv2.putText(after_area, area_text, text_position, 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
   
   return after_area


def display_volume(input, v, size, bottom):

   bottom = int(bottom)
   
   after_volume = input.copy()

   # Display the area on the image
   font_size = 1.2
   # volume_text = f"Volume: {volume:.2f}px3"
   volume_text = f"Volume: {v:.3f} cm3"
   # (text_width, _), _ = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   # text_position = (int(size / 2 - text_width / 2), size - 10)
   (text_width, text_height), text_bottom = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
   text_position = (int(size / 2 - text_width / 2), bottom + 50 + text_height + text_bottom + 10 + text_height + text_bottom)
   cv2.putText(after_volume, volume_text, text_position, 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
   
   return after_volume


def display(plant_id, region_id):

   input_path = f'after_pose/plant{plant_id}_region{region_id}_after3_rotate.jpg'

   with Image.open(input_path) as img:
      size, _ = img.size

   w = read(f'{plant_id}-{region_id}', 'Wpredict')
   h = read(f'{plant_id}-{region_id}', 'Hpredict')
   a = read(f'{plant_id}-{region_id}', 'Apredict')
   v = read(f'{plant_id}-{region_id}', 'Vpredict')

   # Mask
   center_xy = np.load(f'after_seg/plant{plant_id}_region{region_id}_seg_result.npy')
   output = make_mask(input_path, center_xy)
   # cv2.imwrite(f'output/plant{plant_id}_region{region_id}_output1_mask.png', output)

   top = center_xy[:, 1].min()  
   bottom = center_xy[:, 1].max()  
   left = center_xy[:, 0].min() 
   right = center_xy[:, 0].max()  
   
   # Width and Height
   output = display_width_height(output, top, bottom, left, right, w, h)
   # cv2.imwrite(f'output/plant{plant_id}_region{region_id}_output2_width_height.png', output)

   # Area
   output = display_area(output, a, size, bottom)
   # cv2.imwrite(f'output/plant{plant_id}_region{region_id}_output3_area.png', output)

   # Volume
   output = display_volume(output, v, size, bottom)
   cv2.imwrite(f'output/plant{plant_id}_region{region_id}_output4_volume.png', output)


if __name__ == '__main__':

   display(31, 1)