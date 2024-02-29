import matplotlib.pyplot as plt
import numpy as np

# epoch,acc,loss,val_acc,val_loss
#x_axis_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
x_axis_data = []
for i in range(1, 57):
    x_axis_data.append(i)
y_axis_data1 = [82,82.3,82.7,82.5,83.1,83.3,83.5,84,84,84.2,
                84.2,84.2,84.3,84.2,84.6,84.8,84.4,84.2,84.6,84.4,
                84.2,84.4,84.1,84.2,84.1,84.3,84.2,84.3,84.4,84.1,
                84.3,84.2,84.1,84.7,84.3,84.4,84.2,84,84.2,84.4,
                83.5,84.1,84,84.1,84,84.3,84.1,84.3,84.1,84.2,
                84,83.9,84.2,84.2,84,83.7]
y_axis_data2 = [81.7 ,80.4 ,81.3 ,81.8 ,82.3 ,82.6 ,82.0 ,82.3 ,82.3 ,82.4
                ,81.9 ,82.2 ,81.8 ,82.3 ,82.0 ,81.7 ,81.4 ,82.1 ,82.7 ,81.8
                ,82.6 ,82.5 ,82.9 ,82.8 ,81.9 ,82.1 ,82.0 ,81.8 ,82.4 ,82.1
                ,82.3 ,82.4 ,81.9 ,81.9 ,81.5 ,82.2 ,81.9 ,81.0 ,81.8 ,81.9
                ,82.3 ,81.2 ,82.1 ,81.9 ,81.8 ,80.6 ,81.5 ,80.6 ,81.4 ,81.0
                ,81.4 ,81.5 ,81.7 ,81.5 ,80.5 ,81.5]
y_axis_data3 = [82.2 ,82.3 ,82.3 ,82.5 ,82.8 ,82.8 ,83.6 ,83.6 ,83.7 ,84.0
                ,83.8 ,84.1 ,84.0 ,83.8 ,83.9 ,83.9 ,83.9 ,84.0 ,84.2 ,84.1
                ,84.0 ,83.7 ,83.8 ,83.8 ,84.0 ,83.9 ,84.1 ,84.1 ,84.2 ,84.1
                ,84.1 ,84.2 ,84.3 ,84.0 ,84.2 ,84.1 ,84.1 ,83.9 ,84.2 ,84.2
                ,84.4 ,83.8 ,83.9 ,84.2 ,84.0 ,84.0 ,83.9 ,84.2 ,84.0 ,84.0
                ,84.1 ,83.9 ,83.9 ,83.8 ,84.1 ,84.0]
y_axis_data4 = [82.2 ,82.4 ,82.2 ,82.8 ,83.1 ,83.2 ,83.3 ,83.5 ,83.6 ,83.8
                ,83.7 ,83.7 ,84.0 ,84.0 ,84.0 ,84.4 ,84.4 ,83.9 ,84.4 ,84.2
                ,84.2 ,84.0 ,84.1 ,84.2 ,84.0 ,84.2 ,84.4 ,83.7 ,83.9 ,83.9
                ,84.2 ,84.1 ,84.3 ,84.0 ,84.3 ,84.1 ,84.1 ,84.0 ,84.2 ,84.2
                ,84.1 ,84.2 ,84.1 ,83.7 ,84.5 ,83.9 ,84.1 ,84.0 ,84.6 ,84.1
                ,84.6 ,84.3 ,84.2 ,84.4 ,84.4 ,84.2]
y_axis_data5 = [82.3, 82.4, 82.5, 82.3, 82.8, 83.5, 83.1, 83.7, 83.6, 83.6,
                83.8, 84.0, 84.0, 83.9, 83.8, 83.9, 84.1, 84.1, 84.4, 84.0,
                83.9, 84.1, 84.5, 83.9, 84.1, 84.2, 84.6, 84.0, 83.6, 84.4,
                84.0, 84.3, 84.3, 84.1, 84.4, 84.5, 84.5, 84.5, 84.7, 84.3,
                84.3, 84.2, 84.2, 84.0, 84.3, 84.4, 84.3, 84.1, 84.4, 84.1,
                84.0, 84.5, 84.4, 84.2, 83.9, 84.1]
plt.figure(num=0,figsize=(20,8))
# 画图
plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='0')  # '
plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='0.1')
plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='0.2')
plt.plot(x_axis_data, y_axis_data4, 'c*--', alpha=0.5, linewidth=1, label='0.3')  # '
plt.plot(x_axis_data, y_axis_data5, 'm*--', alpha=0.5, linewidth=1, label='0.4')
'''plt.plot(x_axis_data, y_axis_data6, 'ys--', alpha=0.5, linewidth=1, label='1e-7')
plt.plot(x_axis_data, y_axis_data7, 'ko--', alpha=0.5, linewidth=1, label='1e-8')
plt.plot(x_axis_data, y_axis_data8, 'ks--', alpha=0.5, linewidth=1, label='64')'''
## 设置数据标签位置及大小
for a, b in zip(x_axis_data, y_axis_data1):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
for a, b1 in zip(x_axis_data, y_axis_data2):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
for a, b2 in zip(x_axis_data, y_axis_data3):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
for a, b3 in zip(x_axis_data, y_axis_data4):
    plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
for a, b4 in zip(x_axis_data, y_axis_data5):
    plt.text(a, b4, str(b4), ha='center', va='bottom', fontsize=8)
'''for a, b5 in zip(x_axis_data, y_axis_data6):
    plt.text(a, b5, str(b5), ha='center', va='bottom', fontsize=8)
for a, b6 in zip(x_axis_data, y_axis_data7):
    plt.text(a, b6, str(b6), ha='center', va='bottom', fontsize=8)
for a, b7 in zip(x_axis_data, y_axis_data8):
    plt.text(a, b7, str(b7), ha='center', va='bottom', fontsize=8)'''
plt.legend()  # 显示上面的label

plt.xlabel('epoch')
plt.ylabel('accuracy')  # accuracy
# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

