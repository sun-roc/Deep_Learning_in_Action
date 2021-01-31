# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
# 主函数 -- Selective Search
# scale：图像分割的集群程度。值越大，意味集群程度越高，分割的越少，获得子区域越大。默认为1
# sigma: 图像分割前，会先对原图像进行高斯滤波去噪，sigma即为高斯核的大小。默认为0.8
# min_size  : 最小的区域像素点个数。当小于此值时，图像分割的计算就停止，默认为20

# 函数执行的主要流程可以概括为:
# 每次选出相似度最高的一组区域（如编号为100和101的区域），进行合并，得到新的区域（编号为200）。
# 后计算新的区域200与区域100的所有邻居和区域101的所有邻居的相似度，加入区域集S。不断循环，直到S为空，
# 此时最后只剩然下一个区域，而且它的像素数会非常大，接近原始图片的像素数，因此无法继续合并。最后退出程序。

def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50): # img, scale=500, sigma=0.9, min_size=10
    """
    返回值
    -------
        img : ndarray
            包含区域标签的图像
            区域标签存储在每个像素的第四个值 [r,g,b,(region)]
        regions : 字典数组
            [
                {
                    'rect': (left, top, right, bottom),
                    'labels': [...]
                },
                ...
            ]
    """
    #当图片不是三通道时，引发异常
    assert im_orig.shape[2] == 3, "输入非三通道图像"
    # 步骤1：加载图像felzenszwalb获取最小区域
    # 区域标签存储在像素第四维度 [r,g,b,(region)]
    img = add_region_channel(im_orig, scale, sigma, min_size)

    # 图像大小 512 * 512
    img_size = img.shape[0] * img.shape[1] 

    # 步骤2: 将初始分割区域的特征提取出来
    # 每个key下的字典中min_x,min_y,max_x,max_y,labels,size,hist_c,hist_t
    R = get_regions(img)

    # 步骤4: 提取相邻区域
    # 返回一个列表，列表中的每个元素以(a,b)形式，a和b分别是两个有重叠区域的key的字典
    neighbours = get_region_neighbours(R)

    S = {} # 相似度集

    # 步骤5: 衡量相似度
    # ai,bi是region label(0-285)，ar,br是其对应的矩阵
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = sum__sim(ar, br, img_size) # 计算相似度，(ai,bi)对应一个相似度值

    # 步骤六: 合并相似度高的区域
    while S != {}:
        # 步骤6.1：获取最大相似度
        # 获得相似度最高的两个区域标签
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # 步骤6.2：合并区域
        # 开辟一个新的key，存放合并两个最相似区域后的区域
        # 合并区域后新的区域键值
        t = max(R.keys()) + 1.0
        # 步骤6.3：合并区域，放入区域集合
        R[t] = merge_regions(R[i], R[j])
        # 步骤6.4：移除相关区域的相似度
        # 把合并的两个区域的标签，加入待删除列表
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        # 从S里面移除所有关于合并的两个区域的相似度
        for k in key_to_delete:
            del S[k]
        # 步骤6.5：计算新区域与相邻区域相似度
        # 计算新形成的区域的相似度，更新相似度集
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            # 步骤4.6：计算新区域与相邻区域的相似度，放入相似度集合
            S[(t, n)] = sum__sim(R[t], R[n], img_size)

    # 从所有的区域R中抽取目标定位框L，放到新的列表中，返回
    # 步骤5： 提取提议框 [left, top, w, h]
    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions

#1. 用户生成原始区域集的函数，其中用到了felzenszwalb图像分割算法。每一个区域都有一个编号，将编号并入图片中
def add_region_channel(im, scale, sigma, min_size): # img, scale=500, sigma=0.9, min_size=10
    """
        rerurn: 512*512*4的图
    """
    # 产生一层区域的mask
    #计算Felsenszwalb的基于有效图的图像分割。
    #im_mask对每一个像素都进行编号
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im), scale=scale, sigma=sigma,
        min_size=min_size)
    add_channel = np.zeros(im.shape[:2])[:, :, np.newaxis]
    # 把掩码通道合并成图像的第四通道
    im = np.append(im, add_channel, axis=2)
    im[:, :, 3] = im_mask

    return im

#2. 提取区域的尺寸，颜色和纹理特征
def get_regions(img):
    """
        从图像中提取区域，包括区域的尺寸，颜色和纹理特征
        return: 包含min_x,min_y,max_x,max_y,labels,size,hist_c,hist_t这些key的区域字典
    """
    R = {} # 候选区域列表，R的key是区域四个点的

    # 将rgb空间转为hsv空间
    hsv = skimage.color.rgb2hsv(img[:, :, :3]) 

    # 计算区域位置、角点坐标
    for y, i in enumerate(img): # 遍历,img是(x,y,(r,g,b,l))
        for x, (r, g, b, l) in enumerate(i): # 遍历l，从0到285
            # 将所有分割区域的外框加到候选区域列表中
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff, # 把min先设成最大，max先设成最小
                    "max_x": 0, "max_y": 0, "labels": [l]}
            # 更新边界
            if R[l]["min_x"] > x: # 新的x比原来x的最小值更小
                R[l]["min_x"] = x # x的最小值更新为新的x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
    # 计算图像纹理梯度
    tex_grad = LBP_texture(img)

    # 计算每个区域的颜色直方图
     #k是种类，v是此类别的minx,maxx,miny,maxy
    for k, v in list(R.items()): # R中的每一组key, value

        masked_pixels = hsv[:, :, :][img[:, :, 3] == k] # 找出某一key对应区域所有点的h,s,v值
        R[k]["size"] = len(masked_pixels / 4) # 某一key对应区域所有点的个数
        R[k]["hist_c"] = get_color_hist(masked_pixels) # 颜色直方图
        R[k]["hist_t"] = get_texture_hist(tex_grad[:, :][img[:, :, 3] == k]) # 纹理直方图
    # 新增了size,hist_c,hist_t这些key
    return R

# 3. 计算颜色直方图
# 颜色直方图：将色彩空间转为HSV，每个通道下以bins=25计算直方图，这样每个区域的颜色直方图有25*3=75个区间。 

# 纹理相似度：论文采用方差为1的高斯分布在8个方向做梯度统计，然后将统计结果（尺寸与区域大小一致）以bins=10计算直方图。直方图区间数为8*3*10=240（使用RGB色彩空间）。这里是用了LBP获取纹理特征，建立直方图，其余相同
def get_color_hist(img):
    """
        计算输入区域的颜色直方图
        return: BINS * COLOUR_CHANNELS(3)
    """
    #定义BINS数量为25
    BINS = 25
    hist = np.array([])
    # 依次提取每个颜色通道
    for colour_channel in (0, 1, 2):
        # 将输入的参数img各个像素带的第1，2，3hsv色道值提取出来，所以c数组是一维的，c的长度和img是相同的
        c = img[:, colour_channel]
        # numpy.concatenate是拼接函数，将两个函数拼接起来
        # numpy.histogram是计算数据的直方图，即统计哪个数据段中有多少数据，第一个参数是数据矩阵，第二个参数bins指定统计的区间个数，第三个参数是统计的最大最小值
        # 然后将这个类别的三个色道的直方统计拼接在一起
        # 计算每个颜色的直方图，加入到结果中
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
    # L1正则化,得到三个色道的颜色直方图
    hist = hist / len(img)
    return hist
#计算纹理梯度
def LBP_texture(img):
    """
        用LBP(局部二值模式)计算整幅图的纹理梯度,提取纹理特征
        return: 512*512*4
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    # 512*512*4
    return ret
#计算纹理直方图
def get_texture_hist(img):
    """
        计算每个区域的纹理直方图
        输出直方图的大小：BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10
    hist = np.array([])
    for colour_channel in (0, 1, 2):

        # mask by the colour channel
        fd = img[:, colour_channel]

        # 计算每个方向的直方图，加入到结果中
        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    # 标准化
    hist = hist / len(img)
    return hist

#4. 提取相邻区域 通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居参数regions：R记录了该图像每个类别的信息：mix_x,min_y,max_x,max_y,size,hist_c,hist_t
def get_region_neighbours(regions):
    # 检测a,b长方形区域是否存在交叉重叠部分
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False
    # items()取regions的每个元素，即每个类别的信息
    R_list = list(regions.items()) # 把传进来的R以列表形式表示

    neighbours = []
    for cur, a in enumerate(R_list[:-1]):
        for b in R_list[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours

#5. 计算两个区域的相似度
# 论文中考虑了四种相似度 -- 颜色，纹理，尺寸，以及交叠。
# 最后的相似度是四种相似度的加和
def cal_color_sim(r1, r2):
    """
        计算两个区域颜色的相似度
    """
    """
    a = [1,2,3]
    b = [4,5,6]
    zipped = zip(a,b)     # 打包为元组的列表
    [(1, 4), (2, 5), (3, 6)]
    """
    zipped = zip(r1["hist_c"], r2["hist_c"])
    min_list = [min(a, b) for a, b in zipped]
    S_color = sum(min_list)
    return S_color


def cal_texture_sim(r1, r2):
    """
        计算两个区域纹理的相似度
    """
    zipped = zip(r1["hist_t"], r2["hist_t"])
    min_list = S_texture = sum([min(a, b) for a, b in zipped])
    return S_texture


def cal_size_sim(r1, r2, img_size):
    """
        计算两个区域尺寸的相似度
    """
    S_size = 1.0 - (r1["size"] + r2["size"]) / img_size
    return S_size


def cal_fill_sim(r1, r2, img_size):
    """
        计算两个区域交叠的相似度
    """
    Bx = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
    By = (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    # 能包含两个区域的最小矩形区域
    BBsize = (Bx*By)
    S_fill = 1.0 - (BBsize - r1["size"] - r2["size"]) / img_size
    return S_fill


def sum__sim(r1, r2, img_size):
    """
        计算两个区域的相似度
    """
    # 计算类别的相似度
    S_similar = cal_color_sim(r1, r2) + cal_texture_sim(r1, r2)+ cal_size_sim(r1, r2, img_size) + cal_fill_sim(r1, r2, img_size)
    return (S_similar)

# 步骤六合并相似度高的区域
def merge_regions(r1, r2):
    """
        input: 区域字典R中的两个key，也就是两个区域
        output: 合并后的新的区域，代表一个新的key
    """
    new_size = r1["size"] + r2["size"]
    # 合并后的新的区域字典
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt





img = skimage.data.astronaut()
selective_search(img, scale=500, sigma=0.9, min_size=10)