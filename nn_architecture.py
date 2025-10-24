from .config import MIXED_PRECISION

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, Add, Concatenate,
    UpSampling2D, Layer, Lambda, MaxPooling2D)
from tensorflow.keras.models import Model


# ---------------- Mish activation ----------------
class Mish(Layer):
    def call(self, x):
        if MIXED_PRECISION:
            # Выполняем в float32 для стабильности, затем возвращаем в compute_dtype
            x = tf.cast(x, tf.float32)
            out = x * tf.math.tanh(tf.math.softplus(x))
            return tf.cast(out, self.compute_dtype)
        else:
            out = x * tf.math.tanh(tf.math.softplus(x))
            return out


# ---------------- SIMAM attention ----------------
class SimAM(Layer):
    def __init__(self, lam=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def call(self, x):
        if MIXED_PRECISION:
            # Критичные операции в float32
            x32 = tf.cast(x, tf.float32)
            mu = tf.reduce_mean(x32, axis=(1, 2), keepdims=True)
            sigma2 = tf.reduce_mean(tf.square(x32 - mu), axis=(1, 2), keepdims=True)
            w = tf.sigmoid(1. - tf.square(x32 - mu) / (sigma2 + self.lam))
            out = x32 + x32 * w
            return tf.cast(out, self.compute_dtype)
        else:
            mu = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
            sigma2 = tf.reduce_mean(tf.square(x - mu), axis=(1, 2), keepdims=True)
            w = tf.sigmoid(1. - tf.square(x - mu) / (sigma2 + self.lam))
            out = x + x * w
            return out


# ---------------- Ghost Module ----------------
class GhostModule(Layer):
    def __init__(self, out_channels, ratio=2, kernel_size=1, dw_kernel_size=3,
                 use_bn=True, strides=1, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.use_bn = use_bn
        self.strides = strides

        self.intrinsic_c = int(np.ceil(out_channels / ratio))
        self.ghost_c = out_channels - self.intrinsic_c

        self.primary_conv = Conv2D(self.intrinsic_c, kernel_size=kernel_size,
                                   strides=self.strides, padding='same', use_bias=False)
        self.primary_bn = BatchNormalization(dtype='float32') if use_bn else None
        self.mish1 = Mish()

        self.ghost_dw = DepthwiseConv2D(dw_kernel_size, padding='same',
                                        depth_multiplier=1, use_bias=False)
        self.ghost_bn = BatchNormalization(dtype='float32') if use_bn else None
        self.mish2 = Mish()

    def call(self, x):
        x_primary = self.primary_conv(x)
        if self.primary_bn:
            x_primary = self.primary_bn(x_primary)
        x_primary = self.mish1(x_primary)

        x_ghost = self.ghost_dw(x_primary)
        if self.ghost_bn:
            x_ghost = self.ghost_bn(x_ghost)
        x_ghost = self.mish2(x_ghost)

        if self.ghost_c > 0:
            x_ghost = Lambda(lambda z: z[:, :, :, :self.ghost_c])(x_ghost)

        return Concatenate()([x_primary, x_ghost])


# ---------------- Residual block with Ghost ----------------
def ghost_residual_block(x, filters, downsample=False, name=None):
    shortcut = x
    if downsample:
        shortcut = Conv2D(filters, kernel_size=1, strides=2, padding='same',
                          use_bias=False)(shortcut)
        shortcut = BatchNormalization(dtype='float32')(shortcut)

    y = GhostModule(filters, kernel_size=3, dw_kernel_size=3,
                    use_bn=True, strides=2 if downsample else 1,
                    name=f'{name}_ghost1')(x)
    y = GhostModule(filters, kernel_size=3, dw_kernel_size=3,
                    use_bn=True, strides=1,
                    name=f'{name}_ghost2')(y)
    y = BatchNormalization(dtype='float32')(y)

    out = Add()([shortcut, y])
    return out


# ---------------- Encoder ----------------
def build_encoder(inp, base_filters=64):
    x = Conv2D(base_filters, 7, strides=2, padding='same', use_bias=False)(inp)
    x = BatchNormalization(dtype='float32')(x)
    x = Mish()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = ghost_residual_block(x, base_filters, downsample=False, name='res2a')
    x = ghost_residual_block(x, base_filters, downsample=False, name='res2b')
    x = SimAM()(x)
    skip2 = x

    x = ghost_residual_block(x, base_filters * 2, downsample=True, name='res3a')
    x = ghost_residual_block(x, base_filters * 2, downsample=False, name='res3b')
    x = SimAM()(x)
    skip3 = x

    x = ghost_residual_block(x, base_filters * 4, downsample=True, name='res4a')
    x = ghost_residual_block(x, base_filters * 4, downsample=False, name='res4b')
    x = SimAM()(x)
    skip4 = x

    x = ghost_residual_block(x, base_filters * 8, downsample=True, name='res5a')
    x = ghost_residual_block(x, base_filters * 8, downsample=False, name='res5b')
    x = SimAM()(x)

    return x, (skip4, skip3, skip2)


# ---------------- Decoder ----------------
def build_decoder(encoded, skips, base_filters=64):
    skip4, skip3, skip2 = skips

    # upsample 1
    x = UpSampling2D(2, interpolation='bilinear')(encoded)
    x = Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]))([x, skip4])
    x = Concatenate()([x, skip4])
    x = GhostModule(base_filters * 4, kernel_size=3, dw_kernel_size=3)(x)
    x = SimAM()(x)

    # upsample 2
    x = UpSampling2D(2, interpolation='bilinear')(x)
    x = Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]))([x, skip3])
    x = Concatenate()([x, skip3])
    x = GhostModule(base_filters * 2, kernel_size=3, dw_kernel_size=3)(x)
    x = SimAM()(x)

    # upsample 3
    x = UpSampling2D(2, interpolation='bilinear')(x)
    x = Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]))([x, skip2])
    x = Concatenate()([x, skip2])
    x = GhostModule(base_filters, kernel_size=3, dw_kernel_size=3)(x)
    x = SimAM()(x)

    # final upsampling
    x = UpSampling2D(2, interpolation='bilinear')(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)

    return x


# ---------------- Full RGS-UNet model ----------------
def build_rgs_unet(input_shape=(736, 1280, 3), num_classes=1):
    inp = Input(shape=input_shape)
    encoded, skips = build_encoder(inp, base_filters=64)
    decoded = build_decoder(encoded, skips, base_filters=64)
    # Итоговый выход всегда в float32 для корректной работы loss/metrics
    output = Conv2D(num_classes, 1, activation='sigmoid', dtype='float32')(decoded)
    return Model(inputs=inp, outputs=output, name='RGS_UNet')