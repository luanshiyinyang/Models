"""
Author: Zhou Chen
Date: 2020/4/16
Desc: desc
"""
import os
import glob
import shutil
import random
import tqdm


data_folder = '../data/Caltech101/'
train_folder = '../data/Caltech101/train/'
valid_folder = '../data/Caltech101/valid/'


def check_folder():
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(valid_folder):
        os.mkdir(valid_folder)


def split_dataset():
    raw_data_folder = '../data/101_ObjectCategories/'
    categories = os.listdir(raw_data_folder)
    extend_folder = 'BACKGROUND_Google'
    if extend_folder in categories:
        categories.remove(extend_folder)
    print(len(categories))
    label_list = []
    for category in tqdm.tqdm(categories):
        label = categories.index(category)
        label_list.append(label)
        category_folder = os.path.join(raw_data_folder, category)
        files = glob.glob(category_folder + '/*.jpg')
        random.shuffle(files)
        train_size = int(0.8 * len(files))
        train_files = files[:train_size]
        valid_files = files[train_size:]
        out_path = os.path.join(train_folder, str(label))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for img in train_files:
            shutil.copy(img, os.path.join(out_path, os.path.split(img)[-1]))
        out_path = os.path.join(valid_folder, str(label))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for img in valid_files:
            shutil.copy(img, os.path.join(out_path, os.path.split(img)[-1]))


if __name__ == '__main__':
    check_folder()
    split_dataset()