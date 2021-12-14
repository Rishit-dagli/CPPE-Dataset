from typing import List

import tensorflow as tf


def parse_tfrecord_fn(example) -> dict:
    feature_description = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.VarLenFeature(tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image/encoded"] = tf.io.decode_png(example["image/encoded"], channels=3)
    return example


def data_loader(
    record_file_pattern: List[str] = [
        "tfrecords/train.record-00000-of-00001",
        "tfrecords/testdev.record-00000-of-00001",
    ]
):
    dataset = tf.data.TFRecordDataset(record_file_pattern)
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset
