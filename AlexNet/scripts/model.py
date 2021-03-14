# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/7/8 17:16
   desc: the project
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


def AlexNet(input_shape=(224, 224, 3), n_classes=1000):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_tensor = Input(shape=input_shape)
    # conv1
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(input_tensor)
    # x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    # conv2
    x = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    # conv3
    x = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(input_tensor, x)
    return model
