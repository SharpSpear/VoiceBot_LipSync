import random
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import (
    Input,
    Add,
    PReLU,
    Conv2DTranspose,
    Concatenate,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
)
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
from PIL import Image
from keras.utils import Sequence

from tcvc.util import get_image_file_paths


class NoisyImageGenerator(Sequence):
    def __init__(
        self,
        image_dir,
        target_image_dir,
        source_noise_model,
        target_noise_model,
        batch_size=32,
        image_size=128,
    ):
        self.input_image_paths = get_image_file_paths(image_dir)
        self.target_image_paths = get_image_file_paths(target_image_dir)
        assert len(self.input_image_paths) == len(self.target_image_paths)
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.input_image_paths)
        self.batch_size = batch_size
        self.image_size = image_size


    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_idx = random.randint(0, self.image_num - 1)
            input_image_path = self.input_image_paths[image_idx]
            target_image_path = self.target_image_paths[image_idx]
            image = np.array(Image.open(str(input_image_path)))
            target_image = np.array(Image.open(str(target_image_path)))
            assert image.shape == target_image.shape
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_input_patch = image[i : i + image_size, j : j + image_size]
                clean_target_patch = target_image[
                    i : i + image_size, j : j + image_size
                ]
                x[sample_id] = self.source_noise_model(clean_input_patch)
                y[sample_id] = self.target_noise_model(clean_target_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(
        self, image_dir, target_image_dir, val_noise_model, max_images_to_load=2000
    ):
        self.input_image_paths = get_image_file_paths(image_dir)
        self.target_image_paths = get_image_file_paths(target_image_dir)
        assert len(self.input_image_paths) == len(self.target_image_paths)
        self.image_num = len(self.input_image_paths)
        self.approve_rate = 1.0
        if self.image_num > max_images_to_load:
            self.approve_rate = max_images_to_load / self.image_num
        self.data = []


        for i in range(len(self.input_image_paths)):
            if len(self.data) > 1 and random.random() > self.approve_rate:
                continue
            if len(self.data) >= max_images_to_load:
                break
            input_path = self.input_image_paths[i]
            target_path = self.target_image_paths[i]
            x = np.array(Image.open(str(input_path)))
            y = np.array(Image.open(str(target_path)))
            h, w, _ = y.shape
            y = y[: (h // 16) * 16, : (w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(x)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.0)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss

        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print(
                "\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s."
                % (epoch + 1, new_gamma)
            )


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model()
    elif model_name == "unet":
        return get_unet_model(out_ch=3)
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


def get_srresnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def get_unet_model(
    input_channel_num=3,
    out_ch=3,
    start_ch=64,
    depth=4,
    inc_rate=2.0,
    activation="relu",
    dropout=0.0,
    batchnorm=False,
    maxpool=True,
    upconv=True,
    residual=True,
):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding="same")(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding="same")(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding="same")(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding="same")(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding="same")(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(
        i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual
    )
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125

