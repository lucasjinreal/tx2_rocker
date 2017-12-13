
# coding: utf-8

# ### Create TRT engine
#
# In this notebook we create and save a TensorRT engine that we can use in deployment. Be sure that your GPU memory has been freed before running this part of the code.

# ### Imports
#
# In this block we import the necessary python packages

# In[1]:


''' Import TensorRT Modules '''
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser


# ### Configurations

# In[2]:


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

    # after using convert-to-uff tool, I find the graph input name is input_2 not input_1
    "input_layer": "input_2",  # name of the input tensor in the TF computational graph
    # name of the output tensorf in the TF conputational graph
    "out_layer": "dense_2/Softmax",
    "output_size": 5,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)

    "test_image_path": "/tmp/data/val/roses"
}


# ## UFF Model for TRT
#
# In the following function we load the frozen model (tensorflow) and we create the UFF engine. In order to
# create it we need to sspecify (among other things) the precision. Note that the function call for half
# and full precision differs only by one argument (

# In[3]:


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)


# In[4]:


def create_and_save_inference_engine():
    # Define network parameters, including inference batch size, name & dimensionality of input/output layers
    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['out_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = 3
    INPUT_H = config['image_dim']
    INPUT_W = config['image_dim']

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(
        config['frozen_model_file'], OUTPUT_LAYERS)
    print('so we got the uff model.')

    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(INPUT_LAYERS[0], (INPUT_C, INPUT_H, INPUT_W), 0)
    parser.register_output(OUTPUT_LAYERS[0])

    print('# here we failed.')

    # Build your TensorRT inference engine
    if config['precision'] == 'fp32':
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            INFERENCE_BATCH_SIZE,
            1 << 20,
            trt.infer.DataType.FLOAT
        )

    elif config['precision'] == 'fp16':
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            INFERENCE_BATCH_SIZE,
            1 << 20,
            trt.infer.DataType.HALF
        )

    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
    save_path = str(config['engine_save_dir']) + "keras_vgg19_b" + \
        str(INFERENCE_BATCH_SIZE) + "_" + str(config['precision']) + ".engine"

    trt.utils.write_engine_to_file(save_path, engine.serialize())

    print("Saved TRT engine to {}".format(save_path))


# In[5]:


create_and_save_inference_engine()
