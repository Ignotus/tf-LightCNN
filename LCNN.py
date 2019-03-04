import tensorflow as tf
import torch
import numpy as np

# Ported Pytorch code
# https://github.com/AlfredXiangWu/LightCNN/blob/master/light_cnn.py

def mfm(x, out_channels,
        kernel_size=3, stride=1, padding=1, name='conv'):
    if len(x.get_shape()) > 2:
        if padding > 0:
            x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        x = tf.layers.conv2d(x, 2*out_channels, kernel_size, strides=(stride, stride), padding='valid',
                             name=name)
        x1 = x[:,:,:,:out_channels]
        x2 = x[:,:,:,out_channels:]
        return tf.maximum(x1, x2)
    else:
        x = tf.layers.dense(x, 2*out_channels, name=name)
        return tf.maximum(x[:,:out_channels], y[:,out_channels:])

def group(x, out_channels, kernel_size, stride, padding, name='group'):
    with tf.variable_scope(name):
        x = mfm(x, x.get_shape()[-1], 1, 1, 0, name='conv_a')
        x = mfm(x, out_channels, kernel_size, stride, padding, name='conv')
    return x

def resblock(x, out_channels, name='resblock'):
    with tf.variable_scope(name):
        out = mfm(x, out_channels, 3, 1, 1, name='conv1')
        out = mfm(out, out_channels, 3, 1, 1, name='conv2')
    return out + x


def pool(x):
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=[2, 2]) +\
        tf.layers.average_pooling2d(x, pool_size=2, strides=[2, 2])
    return x

def make_layer(x, num_blocks, out_channels):
    for i in range(num_blocks):
        x = resblock(x, out_channels, name='%d' % i)

    return x

def lcnn29_v2(x, layers=[1, 2, 3, 4]):
    x = mfm(x, 48, 5, 1, 2, name='conv1')
    x = pool(x)

    # Block 1
    with tf.variable_scope("block1"):
        x = make_layer(x, layers[0], 48)

    x = group(x, 96, 3, 1, 1, name="group1")
    x = pool(x)

    # Block 2
    with tf.variable_scope("block2"):
        x = make_layer(x, layers[1], 96)
    x = group(x, 192, 3, 1, 1, name="group2")
    x = pool(x)

    # Block 3
    with tf.variable_scope("block3"):
        x = make_layer(x, layers[2], 192)
    x = group(x, 128, 3, 1, 1, name="group3")

    # Block 4
    with tf.variable_scope("block4"):
        x = make_layer(x, layers[3], 128)
    x = group(x, 128, 3, 1, 1, name="group4")
    x = pool(x)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 256, name="fc")
    return x

def lcnn29_v2n(x, name='lcnn29', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        return lcnn29_v2(x)

if __name__ == '__main__':
    x = lcnn29_v2(tf.placeholder(tf.float32, [None, 128, 128, 1], 'img'))

    t_vars = dict()
    for v in tf.trainable_variables():
        t_vars[v.name] = v

    assign_ops = []

    checkpoint = torch.load("LightCNN_29Layers_V2_checkpoint.pth.tar")
    for key in checkpoint['state_dict'].keys():
        tf_key = key.replace("module.", "")
        tf_key = tf_key.replace("filter.", "")
        tf_key = tf_key.replace("weight", "kernel")
        tf_key = tf_key.replace(".", "/")
        tf_key = tf_key + ":0"
        if tf_key in t_vars:
            w = checkpoint['state_dict'][key].cpu().numpy()
            if len(w.shape) == 4:
                print(w.shape)
                w = np.transpose(w, (2, 3, 1, 0))
            elif len(w.shape) == 2:
                w = np.transpose(w, (1, 0))
            assign_ops.append(tf.assign(t_vars[tf_key], w))

    with tf.Session() as sess:
        sess.run(assign_ops)

        tf.train.Saver(var_list=list(t_vars.values())).save(sess, "LCNN29V2.ckpt")

