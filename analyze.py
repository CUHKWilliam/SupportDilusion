import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

with open('debug/stats.txt', 'r') as f:
    stats = f.readlines()
for stat in stats:
    plt.scatter(float(stat.split(" ")[0]), float(stat.split(' ')[1]), alpha=0.1, s=10, c='red')
    print(stat)
plt.savefig('stats.png')
