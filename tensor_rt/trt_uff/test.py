from tensorrt.lite import Engine
from PIL import Image
import numpy as np
import os
import functools
import time

import matplotlib.pyplot as plt


PLAN_single = '/tmp/model/keras_vgg19_b1_fp32.engine'  # engine filename for batch size 1
PLAN_half = '/tmp/model/keras_vgg19_b1_fp16.engine'
IMAGE_DIR = '/tmp/data/val/roses'
BATCH_SIZE = 1


def analyze(output_data):
    LABELS=["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    output = output_data.reshape(-1, len(LABELS))
    
    top_classes = [LABELS[idx] for idx in np.argmax(output, axis=1)]
    top_classes_prob = np.amax(output, axis=1)  

    return top_classes, top_classes_prob


def image_to_np_CHW(image): 
    return np.asarray(
        image.resize(
            (224, 224), 
            Image.ANTIALIAS
        )).transpose([2,0,1]).astype(np.float32)


def load_and_preprocess_images():
    file_list = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    images_trt = []
    for f in file_list:
        images_trt.append(image_to_np_CHW(Image.open(os.path.join(IMAGE_DIR, f))))
    # print('# sample image files: \n', '\n'.join(images_trt[0:3]))
        
    images_trt = np.stack(images_trt)
    
    num_batches = int(len(images_trt) / BATCH_SIZE)
    
    images_trt = np.reshape(images_trt[0:num_batches * BATCH_SIZE], [
        num_batches, 
        BATCH_SIZE, 
        images_trt.shape[1],
        images_trt.shape[2],
        images_trt.shape[3]
    ]) 
    
    return images_trt


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc


def load_TRT_engine(plan):
    if os.path.exists(plan):
        engine = Engine(PLAN=plan, postprocessors={"dense_2/Softmax": analyze})
        return engine
    else:
        print('{} not exist.'.format(plan))


engine_single = load_TRT_engine(PLAN_single)
# engine_half = load_TRT_engine(PLAN_half)


images_trt = load_and_preprocess_images()


@timeit
def infer_all_images_trt(engine):
    results = []
    for image in images_trt:
        result = engine.infer(image) 
        results.append(result)
    return results


results_trt_single = infer_all_images_trt(engine_single)
# results_trt_half = infer_all_images_trt(engine_half)


for i in range(len(results_trt_single)):
    plt.imshow(images_trt[i, 0, 0],  cmap='gray')
    plt.show()
    print(results_trt_single[i][0][0][0])
    # print(results_trt_half[i][0][0][0])

