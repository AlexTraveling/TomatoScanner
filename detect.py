from ultralytics import YOLO
from PIL import Image


def detect(image_path, conf):

   with Image.open(image_path) as img:
      size, _ = img.size

   # Load a model
   YOLO11s = YOLO("weights/YOLO11s.pt")

   # Perform object detection on an image
   results = YOLO11s.predict(image_path,
                           save=True,
                           conf=conf,
                           iou=0.5,
                           imgsz=size)

   result = results[0]

   return result


if __name__ == '__main__':

   detect('input/plant-intro.jpg', 0.75)