# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/7/13 16:28
   desc: 实现DenseNet结构
"""
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K


def Conv_Block(input_tensor, filters, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    """
    封装卷积层
    :param input_tensor: 输入张量
    :param filters: 卷积核数目
    :param bottleneck: 是否使用bottleneck
    :param dropout_rate: dropout比率
    :param weight_decay: 权重衰减率
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)

    if bottleneck:
        # 使用bottleneck进行降维
        inter_channel = filters * 4
        x = Conv2D(inter_channel, (1, 1),
                   kernel_initializer='he_normal',
                   padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def Transition_Block(input_tensor, filters, compression_rate, weight_decay=1e-4):
    """
    封装Translation layer
    :param input_tensor: 输入张量
    :param filters: 卷积核数目
    :param compression_rate: 压缩率
    :param weight_decay: 权重衰减率
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(int(filters * compression_rate), (1, 1),
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def Dense_Block(x, nb_layers, filters, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):
    """
    实现核心的dense block
    :param x: 张量
    :param nb_layers: 模型添加的conv_block数目
    :param filters: 卷积核数目
    :param growth_rate: growth rate
    :param bottleneck: 是否加入bottleneck
    :param dropout_rate: dropout比率
    :param weight_decay: 权重衰减
    :param grow_nb_filters: 是否允许核数目增长
    :param return_concat_list: 是否返回feature map 的list
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]

    for i in range(nb_layers):
        cb = Conv_Block(x, filters, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            filters += growth_rate

    if return_concat_list:
        return x, filters, x_list
    else:
        return x, filters


def DenseNet(n_classes=1000, input_shape=(224, 224, 3), include_top=True, nb_dense_block=4, growth_rate=32, nb_filter=64,
             nb_layers_per_block=[6, 12, 24, 16], bottleneck=True, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4,
             subsample_initial_block=True):
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    final_nb_layer = nb_layers_per_block[-1]
    nb_layers = nb_layers_per_block[:-1]

    compression = 1.0 - reduction
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)
    input_tensor = Input(shape=input_shape)
    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(input_tensor)
    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for block_index in range(nb_dense_block - 1):
        x, nb_filter = Dense_Block(x, nb_layers[block_index], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = Transition_Block(x, nb_filter, compression_rate=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    x, nb_filter = Dense_Block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_tensor, x, name='densenet121')

    return model


if __name__ == '__main__':
    densenet121 = DenseNet()
    print(densenet121.summary())