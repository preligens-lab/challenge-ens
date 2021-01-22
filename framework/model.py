"""
U-Net model definition.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as kl


def UNet(input_shape,
         num_classes=10,
         output_activation='softmax',
         num_layers=4):
    """
    Creates a U-Net model (Ronneberger et al 2015)
    Architecture adapted from github.com/karolzak/keras-unet/master/keras_unet/models/satellite_unet.py
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

    # number of filters in a convolution in the contrastive path (constant)
    filters = 64
    # number of filters in a convolution in the dilative path (constant)
    upconv_filters = 96

    conv2d_kwargs = {
        'kernel_size': (3,3),
        'strides': (1,1),
        'padding': 'same',
        'kernel_initializer': 'he_normal'
    }
    conv2d_transpose_kwargs = {
        'kernel_size': (3,3),
        'strides': (2,2),
        'padding': 'same',
        'output_padding': (1,1)
    }
    maxpool2d_kwargs = {
        'pool_size': (2,2),
        'strides': (2,2),
        'padding': 'valid',
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
    # Test
    unet_kwargs = dict(
        input_shape=(256, 256, 4),
        num_classes=10,
        output_activation='softmax',
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print("Summary:")
    print(model.summary())

    input_batch = tf.random.normal((1, 256, 256, 4), name='random_normal_input')
    output = model(input_batch)
    print(output.shape)
