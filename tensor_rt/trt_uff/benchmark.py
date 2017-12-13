"""
this file will test inference speed using original Keras model
and TensorRT float32 and TensorRT float16
"""
from tensorrt.lite import Engine
from PIL import Image
import numpy as np
import os
import functools
import time
import tensorflow as tf

import matplotlib.pyplot as plt


PLAN_single = '/tmp/model/keras_vgg19_b1_fp32.engine'
PLAN_half = '/tmp/model/keras_vgg19_b1_fp16.engine'
MODEL_FILE = '/tmp/model/keras_vgg19_frozen_model.pb'

IMAGE_DIR = '/tmp/data/val/roses'
BATCH_SIZE = 1


def analyze(output_data):
    labels = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    output = output_data.reshape(-1, len(labels))

    top_classes = [labels[idx] for idx in np.argmax(output, axis=1)]
    top_classes_prob = np.amax(output, axis=1)

    return top_classes, top_classes_prob


def image_to_np_chw(image):
    return np.asarray(
        image.resize(
            (224, 224),
            Image.ANTIALIAS
        )).transpose([2, 0, 1]).astype(np.float32)


def load_and_preprocess_images():
    file_list = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    images = []
    for f in file_list:
        images.append(image_to_np_chw(Image.open(os.path.join(IMAGE_DIR, f))))
    images = np.stack(images)
    num_batches = int(len(images) / BATCH_SIZE)
    images = np.reshape(images[0:num_batches * BATCH_SIZE], [
        num_batches,
        BATCH_SIZE,
        images.shape[1],
        images.shape[2],
        images.shape[3]
    ])
    return images


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        tick = time.time()
        retargs = func(*args, **kwargs)
        tok = time.time() - tick
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(tok * 1000)))
        return retargs
    return newfunc


def load_trt_engine(plan):
    if os.path.exists(plan):
        engine = Engine(PLAN=plan, postprocessors={"dense_2/Softmax": analyze})
        return engine
    else:
        print('{} not exist.'.format(plan))


@timeit
def inference_trt_plan_float32():
    tick = time.time()
    engine = load_trt_engine(PLAN_single)
    images = load_and_preprocess_images()
    results = []
    for image in images:
        result = engine.infer(image)
        results.append(result)
    print('# finished predict {} images.'.format(len(results)))
    print('# all finished in {} seconds.'.format(time.time() - tick))


@timeit
def inference_trt_plan_float16():
    tick = time.time()
    engine = load_trt_engine(PLAN_half)
    images = load_and_preprocess_images()
    results = []
    for image in images:
        result = engine.infer(image)
        results.append(result)
    print('# finished predict {} images.'.format(len(results)))
    print('# all finished in {} seconds.'.format(time.time() - tick))


@timeit
def inference_tf_original():
    tick = time.time()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(MODEL_FILE, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_2 = sess.graph.get_tensor_by_name("input_2:0")
            output = sess.graph.get_tensor_by_name("dense_2/Softmax:0")

            images = load_and_preprocess_images()
            ps = []
            for img in images:
                img = np.transpose(img, (0, 3, 2, 1))
                print(img.shape)
                p = sess.run(output, feed_dict={input_2: img})
                ps.append(p)
    print('# finished solve {} images.'.format(len(ps)))
    print('# all finished in {} seconds.'.format(time.time() - tick))


if __name__ == '__main__':
    print('# Run benchmark on TensorRT speed compare to original inference process.')
    inference_trt_plan_float32()
    # inference_trt_plan_float16()
    inference_tf_original()