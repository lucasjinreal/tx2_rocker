
# coding: utf-8

# # Fine tuning and creating TRT Model
# In this notebook we show how to include TensorRT (TRT) 3.0 in a typical deep learning development workflow. The aim is to show you how you can take advantage of TensorRT to dramatically speed up your inference in a simple and straightforward manner.
#
# In this example we will see how to fine tune a VGG19 architecture trained on Imagenet to categorize different kinds of flower in 5 classes. After fine tuning we will test the accuracy of the model and save it in a format that is understandeable by TensorRT.
#
# ## Workflow
#
# In this notebook we explore the "training" aspect of this problem. For this reason we will need to have tensorflow and keras packages installed in addition to TensorRT 3.0 with python interface, UFF and other modules. Referring to the figure that has been shown at the beginning of the webinar, we will be tackling the "training the model" portion of the slide.
#
#
# ### Imports
# In the following we import the necessary python packages for this part of the hands-on session
#

# In[7]:


''' Import Keras Modules '''
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras


# In[8]:


''' Import Tensorflow Modules '''
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib


#
# ### Training configuration
# In the following we specify configuration parameters that are needed for training

# In[9]:


config = {
    # Training params
    "train_data_dir": "/tmp/data/train",  # training data
    "val_data_dir": "/tmp/data/val",  # validation data
    "train_batch_size": 16,  # training batch size
    "epochs": 3,  # number of training epochs
    "num_train_samples": 2936,  # number of training examples
    "num_val_samples": 734,  # number of test examples

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/tmp/model/keras_vgg19_graphdef.pb",
    "frozen_model_file": "/tmp/model/keras_vgg19_frozen_model.pb",
    "snapshot_dir": "/tmp/data/model/snapshot",
    "engine_save_dir": "/tmp/model/",

    # Needed for TensorRT
    "image_dim": 224,  # the image size (square images)
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "input_1",  # name of the input tensor in the TF computational graph
    # name of the output tensorf in the TF conputational graph
    "out_layer": "dense_2/Softmax",
    "output_size": 5,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)

    "test_image_path": "/tmp/data/val/roses"
}


def finetune_and_freeze_model():
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

    # add a global spatial average pooling layer
    x = base_model.output
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a softmax layer -- in this example we have 5 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will finetune
    model = Model(inputs=base_model.input, outputs=predictions)

    # We want to use the convolutional layers from the pretrained
    # VGG19 as feature extractors, so we freeze those layers and exclude
    # them from training and train only the new top layers
    for layer in base_model.layers:
        print(layer.get_config())
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # create data generators for training/validation
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory=config['train_data_dir'],
        target_size=(config['image_dim'], config['image_dim']),
        batch_size=config['train_batch_size']
    )

    val_generator = val_datagen.flow_from_directory(
        directory=config['val_data_dir'],
        target_size=(config['image_dim'], config['image_dim']),
        batch_size=config['train_batch_size']
    )

    # train the model on the new data for a few epochs
    model.fit_generator(
        train_generator,
        steps_per_epoch=config['num_train_samples'] // config['train_batch_size'],
        epochs=config['epochs'],
        validation_data=val_generator,
        validation_steps=config['num_val_samples'] // config['train_batch_size']
    )

    # Now, let's use the Tensorflow backend to get the TF graphdef and frozen graph
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    # save model weights in TF checkpoint
    checkpoint_path = saver.save(
        sess, config['snapshot_dir'], global_step=0, latest_filename='checkpoint_state')

    # remove nodes not needed for inference from graph def
    train_graph = sess.graph
    inference_graph = tf.graph_util.remove_training_nodes(
        train_graph.as_graph_def())

    # write the graph definition to a file.
    # You can view this file to see your network structure and
    # to determine the names of your network's input/output layers.
    graph_io.write_graph(inference_graph, '.', config['graphdef_file'])

    # specify which layer is the output layer for your graph.
    # In this case, we want to specify the softmax layer after our
    # last dense (fully connected) layer.
    out_names = config['out_layer']

    # freeze your inference graph and save it for later! (Tensorflow)
    freeze_graph.freeze_graph(
        config['graphdef_file'],
        '',
        False,
        checkpoint_path,
        out_names,
        "save/restore_all",
        "save/Const:0",
        config['frozen_model_file'],
        False,
        ""
    )


# ### Run fine-tuning
# Fine-tuning will run for the specified number of epochs

# In[12]:


finetune_and_freeze_model()
