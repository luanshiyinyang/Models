# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/6/17 20:35
   desc: the project
"""
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model


def VGG11(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG11的网络结构
    :param input_shape: 输入图片(H, W, C)尊重设计者这里使用224输入
    :param n_classes: 目标类别
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def VGG13(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG13的网络结构
    :param input_shape: 输入图片(H, W, C)尊重设计者这里使用224输入
    :param n_classes: 目标类别
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def VGG16C(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG16C的网络结构（著名的VGG16）
    没有使用Dropout和BN
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (1, 1), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (1, 1), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (1, 1), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def VGG16D(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG16D的网络结构（著名的VGG16）
    没有使用Dropout和BN
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def VGG19(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG16C的网络结构（著名的VGG16）
    没有使用Dropout和BN
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # block5
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


