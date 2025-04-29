import time

from split import split
from pose import pose
from EdgeBoost import EdgeBoost
from seg import seg
from get_depth import get_depth
from fuse import fuse
from display import display

from operate_database import write


if __name__ == '__main__':

   plant_id = 0
   region_id_list = [1, 2]

   # print('--- Split ---')
   # split(plant_id, conf=0.8)
   # time.sleep(1)

   # print('--- Pose ---')
   # pose(plant_id, region_id_list, conf=0.2)
   # time.sleep(1)

   print('--- EdgeBoost ---')
   EdgeBoost(plant_id, region_id_list)
   time.sleep(1)

   print('--- Seg ---')
   for region_id in region_id_list:
      seg(plant_id, region_id)
      time.sleep(1)

   print('--- Depth Estimation ---')
   # depth = get_depth(plant_id, 2000, 2000)
   # for region_id in region_id_list:
   #    write(f'{plant_id}-{region_id}', 'Depth', depth)
   #    time.sleep(0.5)

   print('--- Fuse ---')
   for region_id in region_id_list:
      fuse(plant_id, region_id)
      time.sleep(0.5)

   print('--- Display ---')
   for region_id in region_id_list:
      display(plant_id, region_id)
      time.sleep(0.5)

   print('TomatoScanner runs successfully!')