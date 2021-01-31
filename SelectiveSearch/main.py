# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import time


def main():

    # 导入宇航员图片，原图是512*512*3，第三维是RGB
    # img = skimage.data.astronaut()
    # img = skimage.data.hubble_deep_field()
    img = skimage.data.chelsea()
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=100)
    # region是一个列表，每一个元素是一个字典，存放每一个区域的信息（rect,size,labels三个key）
    temp = set() # set() 函数创建一个无序不重复元素集
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):
            # temp存储了所有的类别编号
            temp.add(img_lbl[i, j, 3])

    print("原始候选区域:",len(temp))  # 286
    print("SS区域",len(regions))  # 570
    # 创建一个新集合并添加所有区域
    region_rect = set()
    for i,r in enumerate(regions) :
  
        x, y, w, h = r['rect']
        if r['size'] < 1000:
            continue
            # 排除扭曲的候选区域边框  即只保留近似正方形的
        # if w / h > 1.3 or h / w > 1.3:
        #     continue
        region_rect.add(r['rect']) 
            

    # 在原图上绘制矩形框
    # 生成1行1列，大小为6*6的一个字图，fig用来生成一个新的图，ax用来控制子图
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 1
    for x, y, w, h in region_rect:
        # print("Region",i,":",regions[i])
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        i+=1

    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    run_time = end_time - start_time
    print("run time =",run_time,"s")