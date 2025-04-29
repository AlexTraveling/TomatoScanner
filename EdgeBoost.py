from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np


def EdgeBoost(plant_id, region_id_list):

   contrast_rate = 1.2      # 1.5
   sharpness_rate = 1.2     # 1.6
   saturation_factor = 1.0  # 1.0

   for region_id in region_id_list:

      image = Image.open(f'after_pose/plant{plant_id}_region{region_id}_after3_rotate.jpg')  # 打开指定路径的图片

      # 1. 对比度增强
      enh_con = ImageEnhance.Contrast(image)
      image_with_contrast = enh_con.enhance(contrast_rate)

      # 2. 锐化增强
      enh_sharp = ImageEnhance.Sharpness(image_with_contrast)
      image_sharpened = enh_sharp.enhance(sharpness_rate)
      # image_sharpened.save(f'after/tomato_{id}_middle.jpeg')

      # saturation
      enhancer = ImageEnhance.Color(image_sharpened)
      enhanced_image = enhancer.enhance(saturation_factor)
      enhanced_image.save(f'after_EdgeBoost/plant{plant_id}_region{region_id}_after3_rotate_EdgeBoost.jpg')

   print('Edge Highlight Success.')  # 打印成功信息
   print(f'contrast: {contrast_rate}')  # 打印对比度增强倍率
   print(f'sharpness: {sharpness_rate}')  # 打印对比度增强倍率
   print(f'saturation: {saturation_factor}')  # 打印对比度增强倍率


if __name__ == '__main__':

   EdgeBoost(0, [1, 2, 3, 4])
