import matplotlib.pyplot as plt
import numpy as np

# epoch,acc,loss,val_acc,val_loss
x_axis_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
y_axis_data1 = [0.76 ,0.75 ,0.74 ,0.75 ,0.74 ,0.73 ,0.72 ,0.72 ,0.71 ,0.70
                ,0.71 ,0.69 ,0.69 ,0.68 ,0.68 ,0.66 ,0.68 ,0.69 ,0.67 ,0.69
                ,0.68 ,0.69 ,0.67 ,0.66 ,0.66 ,0.66 ,0.66 ,0.67 ,0.67 ,0.66
                ,0.65 ,0.66 ,0.65 ,0.64 ,0.65 ,0.64 ,0.63 ,0.65 ,0.66 ,0.65
                 ,0.64 ,0.65 ,0.64 ,0.64 ,0.63 ,0.64 ,0.62 ,0.63]
y_axis_data2 = [0.75 ,0.73 ,0.73 ,0.72 ,0.72 ,0.73 ,0.73 ,0.71 ,0.70 ,0.69
                ,0.69 ,0.68 ,0.70 ,0.68 ,0.69 ,0.66 ,0.67 ,0.67 ,0.67 ,0.67
                ,0.66 ,0.67 ,0.65 ,0.66 ,0.66 ,0.66 ,0.65 ,0.65 ,0.63 ,0.65
                ,0.63 ,0.63 ,0.63 ,0.65 ,0.65 ,0.64 ,0.63 ,0.64 ,0.63 ,0.63
                ,0.63 ,0.63 ,0.64 ,0.62 ,0.61 ,0.63 ,0.62 ,0.63]
y_axis_data3 = [0.76 ,0.73 ,0.76 ,0.73 ,0.73 ,0.72 ,0.73 ,0.72 ,0.69 ,0.71
                ,0.70 ,0.67 ,0.70 ,0.68 ,0.69 ,0.67 ,0.69 ,0.67 ,0.67 ,0.66
                ,0.65 ,0.67 ,0.67 ,0.67 ,0.66 ,0.66 ,0.65 ,0.63 ,0.66 ,0.64
                ,0.65 ,0.65 ,0.65 ,0.65 ,0.64 ,0.65 ,0.63 ,0.64 ,0.64 ,0.65
                ,0.65 ,0.63 ,0.65 ,0.63 ,0.64 ,0.63 ,0.63 ,0.64]
y_axis_data4 = [0.77 ,0.76 ,0.76 ,0.74 ,0.73 ,0.69 ,0.71 ,0.71 ,0.71 ,0.68
                ,0.69 ,0.71 ,0.69 ,0.68 ,0.68 ,0.69 ,0.68 ,0.69 ,0.68 ,0.67
                ,0.67 ,0.67 ,0.66 ,0.67 ,0.64 ,0.64 ,0.64 ,0.67 ,0.65 ,0.65
                ,0.64 ,0.66 ,0.66 ,0.65 ,0.64 ,0.64 ,0.65 ,0.65 ,0.62 ,0.63
                ,0.64 ,0.64 ,0.63 ,0.64 ,0.64 ,0.63 ,0.63 ,0.64]
y_axis_data8 = [0.81,0.81,0.81,0.81,0.82,0.82,0.82,0.81,0.82,0.81,
                0.81,0.81,0.80,0.80,0.80,0.80,0.80,0.80,0.79,0.80,
                0.80,0.79,0.79,0.79,0.78,0.79,0.79,0.78,0.78,0.78,
                0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.77,0.77,0.77,
                0.77,0.77,0.76,0.77,0.76,0.76,0.77,0.76]

plt.figure(num=0,figsize=(20,8))
# 画图
plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='0.1')  # '
plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='0.2')
plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='0.3')
plt.plot(x_axis_data, y_axis_data4, 'c*--', alpha=0.5, linewidth=1, label='0.4')  # '
'''plt.plot(x_axis_data, y_axis_data5, 'm*--', alpha=0.5, linewidth=1, label='1e-7')
plt.plot(x_axis_data, y_axis_data6, 'ys--', alpha=0.5, linewidth=1, label='5e-8')
plt.plot(x_axis_data, y_axis_data7, 'ko--', alpha=0.5, linewidth=1, label='1e-8')'''
plt.plot(x_axis_data, y_axis_data8, 'ks--', alpha=0.5, linewidth=1, label='0')
## 设置数据标签位置及大小
for a, b in zip(x_axis_data, y_axis_data1):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
for a, b1 in zip(x_axis_data, y_axis_data2):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
for a, b2 in zip(x_axis_data, y_axis_data3):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
for a, b3 in zip(x_axis_data, y_axis_data4):
    plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
'''for a, b4 in zip(x_axis_data, y_axis_data5):
    plt.text(a, b4, str(b4), ha='center', va='bottom', fontsize=8)
for a, b5 in zip(x_axis_data, y_axis_data6):
    plt.text(a, b5, str(b5), ha='center', va='bottom', fontsize=8)
for a, b6 in zip(x_axis_data, y_axis_data7):
    plt.text(a, b6, str(b6), ha='center', va='bottom', fontsize=8)'''
for a, b7 in zip(x_axis_data, y_axis_data8):
    plt.text(a, b7, str(b7), ha='center', va='bottom', fontsize=8)
plt.legend()  # 显示上面的label

plt.xlabel('epoch')
plt.ylabel('memorability')  # accuracy
# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
