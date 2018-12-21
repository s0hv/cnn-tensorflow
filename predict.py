import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from six.moves import cPickle
from PIL import Image

# First, pass the path of the image
save_dir = 'model'
with open(os.path.join(save_dir, 'config.cpkl'), 'rb') as f:
    args = cPickle.load(f)

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = image_path
image_size = args.image_size
num_channels = 3
images = []
# Reading the image using OpenCV
im = Image.open(filename)
bg = Image.new(im.mode, im.size, 'black')
bg.paste(im)
bg = bg.resize((image_size, image_size), Image.LINEAR)
image = np.array(bg)[:,:,:3]  # Leave out alpha channel if it exists
images.append(image)
images = np.array(images, dtype=np.float32)
images = np.multiply(images, 1.0/255.0)
# The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

# Step-1: Recreate the network graph. At this step only graph is created.
with tf.Session() as sess:
    path = tf.train.latest_checkpoint(os.path.join(dir_path, save_dir))
    model_nro = path.split('-')[-1]
    #path = path.replace(model_nro, '44000')
    #model_nro = '44000'
    saver = tf.train.import_meta_graph(os.path.join(dir_path, save_dir, f'model.ckpt-{model_nro}.meta'))
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, path)

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    with open(os.path.join(save_dir, 'labels.cpkl'), 'rb') as f:
        labels = cPickle.load(f)

    y_test_images = np.zeros((1, len(labels)))


    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [P1 P2 P3 P4 ... Pn]

    print(result)
    print(labels[np.argmax(result)])
    print(max(result[0]))
