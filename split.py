from ultralytics import YOLO
import cv2
import time
import torch

from detect import detect


def extarct(category, xywhn):

   output = []

   for i in range(len(category)):
      # judge: fruit or carpopodium
      if category[i] == 0:
         # judge: if in the central area
         x = xywhn[i][0]
         y = xywhn[i][1]
         if 0.2 < x < 0.8 and 0.2 < y < 0.8:
            output.append(xywhn[i])
   
   return output


def save(image_path, location, extend):

   image = cv2.imread(image_path)
   height, width, _ = image.shape
   if height == width:
      length = height

   if location[2] > location[3]:
      max_rate = location[2]
   else:
      max_rate = location[3]
   max_rate *= extend

   unit_resolution = 320

   w = int(max_rate * length)
   w = (int(w / unit_resolution) + 1) * unit_resolution
   print(f'resolution: {w} x {w}')

   x = int(location[0] * length - w / 2)
   y = int(location[1] * length - w / 2)

   # x = int((location[0] - max_rate / 2) * length)
   # y = int((location[1] - max_rate / 2) * length)
   # w = int(max_rate * length)
   extracted_region = image[y:y+w, x:x+w]

   return extracted_region


def split(plant_id, conf):

   # Predict
   image_path = f"input/plant{plant_id}.jpg"

   print('【Predict】')
   result = detect(image_path, conf)

   boxes = result.boxes
   category = boxes.cls
   print(category)
   print(f'category length: {len(category)}')
   xywhn = boxes.xywhn
   print(xywhn)
   print(f'xywhn length: {len(xywhn)}')

   # Extract
   print('【Extract】')
   after_extract = extarct(category, xywhn)
   for single in after_extract:
      print(single)
   print(f'xywhn length after extraction: {len(after_extract)}')
   # save extract
   # torch.save(after_extract, 'individual_position.pth')
   # loaded_tensor_list = torch.load('individual_position.pth')

   # Save
   print('【Save】')
   for i in range(len(after_extract)):
      extend = 2.0
      region = save(image_path, after_extract[i], extend)
      cv2.imwrite(f'after_split/plant{plant_id}_region{i + 1}.jpg', region)
      time.sleep(0.5)


if __name__ == '__main__':
   
   split(28, conf=0.8)