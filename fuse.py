import numpy as np
import matplotlib.pyplot as plt

from operate_database import read, write


def display(depth, pixel, coeffs, depth_input, pixel_pred):

   # 生成拟合曲线
   depth_fit = np.linspace(min(depth), max(depth), 100)  # 在深度范围内取100个点
   pixel_fit = coeffs[0] / depth_fit + coeffs[1]  # 计算拟合曲线

   # 绘图
   plt.figure(figsize=(8, 4))
   plt.plot(depth_fit, pixel_fit, color='#006666', label='Pixel-Depth Curve')  # 拟合曲线
   plt.scatter(depth, pixel, color='black', label='Standard data')  # 原始数据点
   plt.scatter(depth_input, pixel_pred, color='#CC3300', s=100, label=f'How many pixels correspond to 1cm: {pixel_pred:.2f}')  # 预测点
   plt.xlabel('Depth / cm')
   plt.ylabel('Pixel')
   # plt.title('Demonstration')
   plt.legend()
   plt.grid()
   plt.savefig("fuse_curve.png", dpi=300, transparent=True)  # 保存图片


def fuse(plant_id, region_id):

   id = f'{plant_id}-{region_id}'

   # 24
   # depth = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
   # pixel = np.array([1532, 1025, 750, 584, 486, 410, 354, 315, 282, 257, 236]) / 10  # 这里除以10

   # 35
   depth = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
   pixel = np.array([2710, 1872, 1420, 1158, 984, 852, 744, 675, 612, 546, 504]) / 10  # 这里除以10

   # 反比例拟合
   coeffs = np.polyfit(1 / depth, pixel, 1)

   # 计算 depth = 35 对应的 pixel 值

   depth_input = read(id, 'Depth')
   pixel_pred = coeffs[0] / depth_input + coeffs[1]
   print(f"when depth = {depth_input} cm, pixel = {pixel_pred:.2f}")

   display(depth, pixel, coeffs, depth_input, pixel_pred)

   w = read(id, 'Wpixel')
   h = read(id, 'Hpixel')
   a = read(id, 'Apixel')
   v = read(id, 'Vpixel')

   w /= pixel_pred
   h /= pixel_pred
   a /= (pixel_pred**2)
   v /= (pixel_pred**3)

   print(f'w: {w:.3f}')
   print(f'h: {h:.3f}')
   print(f'a: {a:.3f}')
   print(f'v: {v:.3f}')

   write(id, 'Wpredict', w)
   write(id, 'Hpredict', h)
   write(id, 'Apredict', a)
   write(id, 'Vpredict', v)


if __name__ == '__main__':

   fuse(0, 1)