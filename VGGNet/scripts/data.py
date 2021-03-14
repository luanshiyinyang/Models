# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/6/23 20:26
   desc: 读取Caltech101数据集
"""


class Caltech101(object):
    def __init__(self):
        self.folder = '../data/101_ObjectCategories/'

    def gen_data(self):
        import os
        import cv2
        import numpy as np
        from tqdm import tqdm
        categories = os.listdir(self.folder)
        # 去除干扰项
        categories.remove('BACKGROUND_Google')
        x_train = []
        y_train = []
        for i in tqdm(range(len(categories))):
            root = os.path.join(self.folder, categories[i])
            for file in os.listdir(root):
                file_name = os.path.join(root, file)
                img = cv2.resize(cv2.imread(file_name), (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return x_train, y_train
