"""
U-Net model definition.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as kl



# def UNetYannLeguilly(input_shape,
#                      num_classes=10,
#                      output_activation='softmax'):
#     """
#     Source: https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
#     """
#     initializer = 'he_normal'

#     # -- Encoder -- #
#     # Block encoder 1
#     inputs = kl.Input(shape=input_shape)
#     conv_enc_1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
#     conv_enc_1 = kl.Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

#     # Block encoder 2
#     max_pool_enc_2 = kl.MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
#     conv_enc_2 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
#     conv_enc_2 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

#     # Block  encoder 3
#     max_pool_enc_3 = kl.MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
#     conv_enc_3 = kl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
#     conv_enc_3 = kl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

#     # Block  encoder 4
#     max_pool_enc_4 = kl.MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
#     conv_enc_4 = kl.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
#     conv_enc_4 = kl.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
#     # -- Encoder -- #

#     # ----------- #
#     maxpool = kl.MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
#     conv = kl.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
#     conv = kl.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
#     # ----------- #

#     # -- Decoder -- #
#     # Block decoder 1
#     up_dec_1 = kl.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(kl.UpSampling2D(size = (2,2))(conv))
#     merge_dec_1 = kl.concatenate([conv_enc_4, up_dec_1], axis = 3)
#     conv_dec_1 = kl.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
#     conv_dec_1 = kl.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

#     # Block decoder 2
#     up_dec_2 = kl.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(kl.UpSampling2D(size = (2,2))(conv_dec_1))
#     merge_dec_2 = kl.concatenate([conv_enc_3, up_dec_2], axis = 3)
#     conv_dec_2 = kl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
#     conv_dec_2 = kl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

#     # Block decoder 3
#     up_dec_3 = kl.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(kl.UpSampling2D(size = (2,2))(conv_dec_2))
#     merge_dec_3 = kl.concatenate([conv_enc_2, up_dec_3], axis = 3)
#     conv_dec_3 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
#     conv_dec_3 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

#     # Block decoder 4
#     up_dec_4 = kl.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(kl.UpSampling2D(size = (2,2))(conv_dec_3))
#     merge_dec_4 = kl.concatenate([conv_enc_1, up_dec_4], axis = 3)
#     conv_dec_4 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
#     conv_dec_4 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
#     conv_dec_4 = kl.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
#     # -- Dencoder -- #

#     outputs = kl.Conv2D(num_classes, 1, activation =output_activation)(conv_dec_4)

#     model = tf.keras.Model(inputs =[inputs], outputs =[outputs])
#     return model


# ----------------------------------------------------------------------------#
# Below:
# Source: karolzak/keras-unet/master/keras_unet/models/satellite_unet.py


def UNet(input_shape,
         num_classes=10,
         output_activation='softmax',
         num_layers=4):
    """Creates a U-Net model (Ronneberger et al 2015)
    @ TODO: docstring
    """

    def bn_conv_relu(input, filters, **conv2d_kwargs):
        x = kl.BatchNormalization()(input)
        x = kl.Conv2D(filters, activation='relu', **conv2d_kwargs)(x)
        return x

    def bn_upconv_relu(input, filters, **conv2d_transpose_kwargs):
        x = kl.BatchNormalization()(input)
        x = kl.Conv2DTranspose(filters, activation='relu', **conv2d_transpose_kwargs)(x)
        return x

    inputs = kl.Input(input_shape)

    # number of filters in a convolution in the contrastive path: constant
    filters = 64
    # number of filters in a convolution in the dilative path: constant
    upconv_filters = 96

    kernel_size = (3,3)
    strides = (1,1)
    padding = 'same'
    kernel_initializer = 'he_normal'
    conv2d_kwargs = {
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }
    conv2d_transpose_kwargs = {
        'kernel_size': kernel_size,
        'strides': (2,2),
        'padding': padding,
        'output_padding': (1,1)
    }

    pool_size = (2,2)
    pool_strides = (2,2)
    pool_padding = 'valid'
    maxpool2d_kwargs = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x = kl.Conv2D(filters, activation='relu', **conv2d_kwargs)(inputs)
    c1 = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_conv_relu(c1, filters, **conv2d_kwargs)
    x = kl.MaxPooling2D(**maxpool2d_kwargs)(x)

    down_layers = []

    for _ in range(num_layers):
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = kl.MaxPooling2D(**maxpool2d_kwargs)(x)

    x = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_upconv_relu(x, filters, **conv2d_transpose_kwargs)

    for conv in reversed(down_layers):
        x = kl.concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, **conv2d_kwargs)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = bn_upconv_relu(x, filters, **conv2d_transpose_kwargs)

    x = kl.concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, **conv2d_kwargs)
    x = bn_conv_relu(x, filters, **conv2d_kwargs)

    outputs = kl.Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)

    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    return model


if __name__ == '__main__':

    unet_kwargs = dict(input_shape=(256, 256, 4), num_classes=10, output_activation='softmax')
    print("Creating U-Net Yann Leguilly with arguments: {unet_kwargs}")
    model = UNetYannLeguilly(**unet_kwargs)
    print("Summary:")
    print(model.summary())

    input_batch = tf.random.normal((1, 256, 256, 4), name='random_normal_input')
    output = model(input_batch)
    print(output.shape)

    unet_kwargs = dict(input_shape=(256, 256, 4), num_classes=10, output_activation='softmax', num_layers=2)
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print("Summary:")
    print(model.summary())

    input_batch = tf.random.normal((1, 256, 256, 4), name='random_normal_input')
    output = model(input_batch)
    print(output.shape)
