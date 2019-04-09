# -*- coding: utf-8 -*-
import os 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference


def create():
    
    x=tf.placeholder(
        tf.float32,
        [1,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
        name='x-input'
    )

    y=mnist_inference.inference(x)

    output=tf.nn.softmax(y,name="y-output")

    saver=tf.train.Saver(tf.global_variables())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
        saver.restore(
                    sess,
                    "./model/cnn-mnist.ckpt-30000")

        saver.save(sess,"./ncs/cnn-mnist_inference")
        print("create successful!")
def main(argv=None):
    create()

if __name__ == "__main__":
    main()