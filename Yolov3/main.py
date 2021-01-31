import codecs
import sys
import numpy as np
import time
import paddle
import paddle.fluid as fluid
import math
import functools

from IPython.display import display
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import namedtuple
import Yolo_v3_train
with open(u"parameter.txt",encoding="utf-8") as f:  # 打开文件
    data = f.read()  #encoding='UTF-8'
    print(type(data),data)
    train_params = eval(data)
    print(train_params)
yolo_config =  train_params['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']
label_dict = train_params['num_dict']
class_dim = train_params['class_dim']
print("label_dict:{} class dim:{}".format(label_dict, class_dim))

place = fluid.CUDAPlace(0) if train_params['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)

path = train_params['freeze_dir']
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


# 给图片画上外接矩形框
def draw_bbox_image(img, boxes, labels, save_name):
    img_width, img_height = img.size

    draw = ImageDraw.Draw(img) # 图像绘制对象
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red') # 绘制矩形
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0)) # 绘制标签
    img.save(save_name)
    display(img)


def resize_img(img, target_size):
    # 保持比例的缩放图片
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    #读取图片
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def test(image_path):
    # 预测，将结果保存到一副新的图片中
    origin, tensor_img, resized_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))

    t1 = time.time()
    # 执行预测
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    bboxes = np.array(batch_outputs[0])  # 预测结果
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return
    labels = bboxes[:, 0].astype('int32') # 类别
    scores = bboxes[:, 1].astype('float32') # 概率
    boxes = bboxes[:, 2:].astype('float32') # 边框

    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-result.jpg'
    draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    image_name = sys.argv[1]
    image_path = image_name
    image_path = "data/2.jpg"
    test(image_path)