# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SEC = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(
            tf.float32,
            [5000,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
            name='x-input'
        )
        y_=tf.placeholder(
            tf.float32,
            [5000,mnist_inference.OUTPUT_NODE],
            name="y-output"
        )
        xs=mnist.validation.images
        print(xs.shape)
        reshaped_xs=np.reshape(xs,(
                xs.shape[0],
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS
            ))

        print(reshaped_xs.shape)
        validate_feed={x: reshaped_xs,
                        y_:mnist.validation.labels}
        
        y=mnist_inference.inference(x,None)

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages=tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    accuracy_score=sess.run(accuracy,
                    feed_dict=validate_feed)
                    print("accuracy=%g"%(accuracy_score))
                else:
                    print("No checkpoint file found")
                    return 
                time.sleep(EVAL_INTERVAL_SEC)

def main(argv=None):
    mnist =input_data.read_data_sets(
    "/path/to/MNIST_data",one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    main()