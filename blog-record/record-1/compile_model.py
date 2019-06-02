import pdb
pdb.set_trace()

import nnvm
import tvm
import os.path
import sys
import numpy as np
import tensorflow as tf

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
map_proto_url = os.path.join(repo_base, map_proto)
label_map = 'imagenet_synset_to_human_label_map.txt'
label_map_url = os.path.join(repo_base, label_map)

# Target settings
target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

# Download required files
from tvm.contrib.download import download_testdata
img_path = download_testdata(image_url, img_name, module='data')
model_path = download_testdata(model_url, model_name, module=['tf', 'InceptionV1'])
map_proto_path = download_testdata(map_proto_url, map_proto, module='data')
label_path = download_testdata(label_map_url, label_map, module='data')

# Import model
import tvm.relay.testing.tf as tf_testing
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

# Decode image
from PIL import Image
image = Image.open(img_path).resize((299, 299))
x = np.array(image)

# Import the graph to NNVM
sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout)
print("Tensorflow protobuf imported as nnvm graph")

# NNVM Compilation
import nnvm.compiler
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
graph, lib, params = nnvm.compiler.build(
    sym, shape=shape_dict, target=target, target_host=target_host,
    dtype=dtype_dict, params=params)

# Save Compiled Module
from tvm.contrib import util
temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print('module saved')
print(temp.listdir())

