import tensorflow as tf
import os
import numpy as np

_HEIGHT_EXTRA_PIXELS = 4
_WIDTH_EXTRA_PIXELS = 4

class DataSet(object):
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir
        self.header = None

    def set_config(self, hwd_shape, label_bytes,
                n_train_images, n_test_images, n_classes):

        self.height, self.width, self.depth = hwd_shape
        self.label_bytes = label_bytes
        self.n_train_images = n_train_images
        self.n_classes = n_classes
        self.n_test_images = n_test_images

        self.stats = {
            'n_test': n_test_images,
            'n_train': n_train_images,
            'n_classes': n_classes
        }

    def _get_hwd_shape(self):
        return (self.height, self.width, self.depth)

    def record_dataset(self, filenames):
        height, width, depth = self._get_hwd_shape()
        record_bytes = height*width*depth + self.label_bytes
        return tf.data.FixedLengthRecordDataset(filenames, record_bytes)

    def parse_record(self, value):
        height, width, depth = self._get_hwd_shape()

        label_bytes = self.label_bytes
        image_bytes = height*width*depth
        record_bytes = label_bytes+image_bytes

        raw_record = tf.decode_raw(value, tf.uint8)
        label = tf.cast(raw_record[0], tf.int32)

        depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
                                   [depth, height, width])
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
        return image, tf.one_hot(label, self.n_classes)

    def preprocess_fn(self, image, label):
        height, width, depth = self._get_hwd_shape()
        image = tf.image.resize_image_with_crop_or_pad(image,
                            height +_HEIGHT_EXTRA_PIXELS,
                            width +_WIDTH_EXTRA_PIXELS)
        image = tf.random_crop(image, [height, width, depth])
        image = tf.image.random_flip_left_right(image)
        return image, label
