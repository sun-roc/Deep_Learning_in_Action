import os
import numpy as np
import time
import math
import random
import paddle
import paddle.fluid as fluid
import codecs
import json

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from PIL import Image, ImageEnhance, ImageDraw

with open(u"parameter.txt",encoding="utf-8") as f:  # 打开文件
    data = f.read()  #encoding='UTF-8'
    print(type(data),data)
    train_params = eval(data)
    print(train_params)


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    """
    file_list = os.path.join("data", "train.txt")  # 训练集
    label_list = os.path.join("data", "label_list")  # 标签文件
    index = 0

    # codecs是专门用作编码转换通用模块
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_params['num_dict'][index] = line.strip()
            train_params['label_dict'][line.strip()] = index
            index += 1
        train_params['class_dim'] = index

    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_params['image_count'] = len(lines)  # 图片数量




# 定义YOLO3网络结构：darknet-53
class YOLOv3(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []  # 网络最终模型
        self.downsample_ratio = 1  # 下采样率
        self.anchor_mask = anchor_mask  
        self.anchors = anchors  # 锚点
        self.class_num = class_num  # 类别数量

        self.yolo_anchors = []
        self.yolo_classes = []

        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    # 获取anchors
    def get_anchors(self):
        return self.anchors

    # 获取anchor_mask
    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    # 卷积正则化函数: 卷积、批量正则化处理、leakrelu
    def conv_bn(self,
                input,  # 输入
                num_filters,  # 卷积核数量
                filter_size,  # 卷积核大小
                stride,  # 步幅
                padding,  # 填充
                use_cudnn=True):
        # 2d卷积操作
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=padding,
                                   act=None,
                                   use_cudnn=use_cudnn,  # 是否使用cudnn
                                   param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                                   bias_attr=False)

        # batch_norm中的参数不需要参与正则化
        # 正则化的目的，是为了防止过拟合，较小的L2值能防止过拟合
        param_attr = ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                               regularizer=L2Decay(0.))
        bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0),
                              regularizer=L2Decay(0.))
        out = fluid.layers.batch_norm(input=conv, act=None,
                                      param_attr=param_attr,
                                      bias_attr=bias_attr)
        # leaky_relu: Leaky ReLU是给所有负值赋予一个非零斜率
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    # 通过卷积实现降采样
    def down_sample(self, input, num_filters, filter_size=3, stride=2, padding=1):
        self.downsample_ratio *= 2  # 降采样率
        return self.conv_bn(input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding)

    # 基本块：包含两个卷积/正则化层，一个残差块
    def basic_block(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        conv2 = self.conv_bn(conv1, num_filters * 2, filter_size=3, stride=1, padding=1)
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)  # 计算H(x)=F(x)+x
        return out

    # 创建多个basic_block
    def layer_warp(self, input, num_filters, count):
        res_out = self.basic_block(input, num_filters)
        for j in range(1, count):
            res_out = self.basic_block(res_out, num_filters)
        return res_out

    # 上采样
    def up_sample(self, input, scale=2):
        shape_nchw = fluid.layers.shape(input)  # 获取input的形状
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale  # 计算输出数据形状
        out_shape.stop_gradient = True

        # 矩阵放大(最邻插值法)
        out = fluid.layers.resize_nearest(input=input,
                                          scale=scale,
                                          actual_shape=out_shape)
        return out

    def yolo_detection_block(self, input, num_filters):

        conv = input
        for j in range(2):
            conv = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_bn(conv, num_filters * 2, filter_size=3, stride=1, padding=1)
        route = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    # 搭建网络模型 darknet-53
    def net(self, img):
        stages = [1, 2, 8, 8, 4]
        # 第一个卷积层: 256*256
        conv1 = self.conv_bn(img, num_filters=32, filter_size=3, stride=1, padding=1)
        # 第二个卷积层：128*128
        downsample_ = self.down_sample(conv1, conv1.shape[1] * 2)  # 第二个参数为卷积核数量
        blocks = []

        # 循环创建basic_block组
        for i, stage_count in enumerate(stages):
            block = self.layer_warp(downsample_,  # 输入数据
                                    32 * (2 ** i),  # 卷积核数量
                                    stage_count)  # 基本块数量
            blocks.append(block)
            if i < len(stages) - 1:  # 如果不是最后一组，做降采样
                downsample_ = self.down_sample(block, block.shape[1] * 2)
        blocks = blocks[-1:-4:-1]  # 取倒数三层，并且逆序，后面跨层级联需要

        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)  # 连接route和block，按行

            route, tip = self.yolo_detection_block(block,  # 输入
                                                   num_filters=512 // (2 ** i))  # 卷积核数量

            param_attr = ParamAttr(initializer=fluid.initializer.Normal(0., 0.02))
            bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.))
            block_out = fluid.layers.conv2d(input=tip,
                                            # 5 elements represent x|y|h|w|score
                                            num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),
                                            filter_size=1,
                                            stride=1,
                                            padding=0,
                                            act=None,
                                            param_attr=param_attr,
                                            bias_attr=bias_attr)
            self.outputs.append(block_out)

            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 256 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.up_sample(route)  # 上采样

        return self.outputs




def get_yolo(class_num, anchors, anchor_mask):
        return YOLOv3(class_num, anchors, anchor_mask)


class Sampler(object):
    """
    采样器，用于扣取采样
    """

    def __init__(self, max_sample, max_trial, min_scale, max_scale,
                 min_aspect_ratio, max_aspect_ratio, min_jaccard_overlap,
                 max_jaccard_overlap):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap


class bbox(object):
    """
    外界矩形框
    """

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


# 坐标转换，由[x1, y1, w, h]转换为[center_x, center_y, w, h]
# 并转换为范围在[0, 1]之间的相对坐标
def box_to_center_relative(box, img_height, img_width):


    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w - 1, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, img_height - 1)

    x = (x1 + x2) / 2 / img_width  # x中心坐标
    y = (y1 + y2) / 2 / img_height  # y中心坐标
    w = (x2 - x1) / img_width  # 框宽度/图片总宽度
    h = (y2 - y1) / img_height  # 框高度/图片总高度

    return np.array([x, y, w, h])


# 调整图像大小
def resize_img(img, sampled_labels, input_size):
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


# 计算交并比
def box_iou_xywh(box1, box2):


    # 取两个框的坐标
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1  # 相交部分宽度
    inter_h = inter_y2 - inter_y1 + 1  # 相交部分高度
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h  # 相交面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  # 框1的面积
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  # 框2的面积

    return inter_area / (b1_area + b2_area - inter_area)  # 相集面积/并集面积


# box裁剪
def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


# 图像增加：对比度，饱和度，明暗，颜色，扩张
def random_brightness(img):  # 亮度
    prob = np.random.uniform(0, 1)

    if prob < train_params['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_params['image_distort_strategy']['brightness_delta']  # 默认值0.125
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1  # 产生均匀分布随机值
        img = ImageEnhance.Brightness(img).enhance(delta)  # 调整图像亮度

    return img


def random_contrast(img):  # 对比度
    prob = np.random.uniform(0, 1)

    if prob < train_params['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_params['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)

    return img


def random_saturation(img):  # 饱和度
    prob = np.random.uniform(0, 1)

    if prob < train_params['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_params['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)

    return img


def random_hue(img):  # 色调
    prob = np.random.uniform(0, 1)

    if prob < train_params['image_distort_strategy']['hue_prob']:
        hue_delta = train_params['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

    return img



# 随机裁剪
def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [(0.1, 1.0),
                       (0.3, 1.0),
                       (0.5, 1.0),
                       (0.7, 1.0),
                       (0.9, 1.0),
                       (0.0, 1.0)]  # 最小/最大交并比值

    w, h = img.size
    crops = [(0, 0, w, h)]

    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h / float(h)
            ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels


# 扩张
def random_expand(img, gtboxes, keep_ratio=True):
    if np.random.uniform(0, 1) < train_params['image_distort_strategy']['expand_prob']:
        return img, gtboxes

    max_ratio = train_params['image_distort_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_params['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes


# 预处理：图像样本增强，维度转换
def preprocess(img, bbox_labels, input_size, mode):
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)

    if mode == 'train':


        img, gtboxes = random_expand(img, sample_labels[:, 1:5])  # 扩展增强
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])  # 随机裁剪
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes

    img = resize_img(img, sample_labels, input_size)
    img = np.array(img).astype('float32')
    img -= train_params['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


# 数据读取器
# 根据样本文件，读取图片、并做数据增强，返回图片数据、边框、标签
def custom_reader(file_list, data_dir, input_size, mode):
    def reader():
        np.random.shuffle(file_list)  # 打乱文件列表

        for line in file_list:  # 读取行，每行一个图片及标注
            if mode == 'train' or mode == 'eval':
                parts = line.split('\t')  # 按照tab键拆分
                image_path = parts[0]

                img = Image.open(os.path.join(data_dir, image_path)) # 读取图像数据
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size

                # bbox 的列表，每一个元素为这样
                bbox_labels = []
                for object_str in parts[1:]:  # 循环处理每一个目标标注信息
                    if len(object_str) <= 1:
                        continue

                    bbox_sample = []
                    object = json.loads(object_str)
                    bbox_sample.append(float(train_params['label_dict'][object['value']]))
                    bbox = object['coordinate']  # 获取框坐标
                    # 计算x,y,w,h
                    box = [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                    bbox = box_to_center_relative(box, im_height, im_width)  # 坐标转换
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    difficult = float(0)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)

                if len(bbox_labels) == 0:
                    continue

                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)  # 预处理
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0:
                    continue

                boxes = sample_labels[:, 1:5]  # 坐标
                lbls = sample_labels[:, 0].astype('int32')  # 标签
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_params['max_box_num']  # 一副图像最多多少个目标物体
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)  # 控制最大目标数量
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]

                yield img, ret_boxes, ret_lbls

            elif mode == 'test':
                img_path = os.path.join(line)

                yield Image.open(img_path)

    return reader


# 批量、随机数据读取器
def single_custom_reader(file_path, data_dir, input_size, mode):
    file_path = os.path.join(data_dir, file_path)

    images = [line.strip() for line in open(file_path)]
    reader = custom_reader(images, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_params['train_batch_size'])
    reader = paddle.batch(reader, train_params['train_batch_size'])

    return reader


# 定义优化器
def optimizer_sgd_setting():
    batch_size = train_params["train_batch_size"]  # batch大小
    iters = train_params["image_count"] // batch_size  # 计算轮次
    iters = 1 if iters < 1 else iters
    learning_strategy = train_params['sgd_strategy']
    lr = learning_strategy['learning_rate']  # 学习率

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.SGDOptimizer(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),  # 分段衰减学习率
        # learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


# 创建program, yolo模型
def build_program_with_feeder(main_prog, startup_prog, place):
    max_box_num = train_params['max_box_num']
    yolo_config =  train_params['yolo_cfg']

    with fluid.program_guard(main_prog, startup_prog):  # 更改全局主程序和启动程序
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')  # 图像
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')  # 边框
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')  # 标签

        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label],
                                  place=place,
                                  program=main_prog)  # 定义feeder
        reader = single_custom_reader("train.txt",
                                      "data",
                                      yolo_config['input_size'], 'train')  # 读取器
        # 获取yolo参数

        yolo_config =  else train_params['yolo_cfg']

        with fluid.unique_name.guard():
            # 创建yolo模型
            model = get_yolo( train_params['class_dim'], yolo_config['anchors'],
                             yolo_config['anchor_mask'])
            outputs = model.net(img)
        return feeder, reader, get_loss(model, outputs, gt_box, gt_label)


# 损失函数
def get_loss(model, outputs, gt_box, gt_label):
    losses = []
    downsample_ratio = model.get_downsample_ratio()

    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            loss = fluid.layers.yolov3_loss(x=out,
                                            gt_box=gt_box,  # 真实边框
                                            gt_label=gt_label,  # 标签
                                            anchors=model.get_anchors(),  # 锚点
                                            anchor_mask=model.get_anchor_mask()[i],
                                            class_num=model.get_class_num(),
                                            ignore_thresh=train_params['ignore_thresh'],
                                            # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                                            use_label_smooth=False,
                                            downsample_ratio=downsample_ratio)
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer = optimizer_sgd_setting()
        optimizer.minimize(loss)
        return loss





# 执行训练
def train():
    init_log_config()
    init_train_parameters()
    print("开始训练")
    place = fluid.CUDAPlace(0) if train_params['use_gpu'] else fluid.CPUPlace()

    train_program = fluid.Program()
    start_program = fluid.Program()
    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)

    print("初始化参数")

    exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss.name]


    stop_strategy = train_params['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']

    stop_train = False
    successive_count = 0
    total_batch_count = 0
    valid_thresh = train_params['valid_thresh']
    nms_thresh = train_params['nms_thresh']
    current_best_loss = 10000000000.0

    # 开始迭代训练
    for pass_id in range(train_params["num_epochs"]):
        batch_id = 0
        total_loss = 0.0

        for batch_id, data in enumerate(reader()):
            t1 = time.time()

            loss = exe.run(train_program,
                           feed=feeder.feed(data),
                           fetch_list=train_fetch_list)  # 执行训练

            period = time.time() - t1
            loss = np.mean(np.array(loss))
            total_loss += loss
            batch_id += 1
            total_batch_count += 1

            if batch_id % 10 == 0:  # 调整日志输出的频率
                print(
                    "pass {}, trainbatch {}, loss {} time {}".format(pass_id, batch_id, loss, "%2.2f sec" % period))

        pass_mean_loss = total_loss / batch_id
        print("pass {0} train result, current pass mean loss: {1}".format(pass_id, pass_mean_loss))

    fluid.io.save_persistables(dirname=train_params['save_model_dir'], main_program=train_program, executor=exe)
if __name__ == '__main__':
    train()