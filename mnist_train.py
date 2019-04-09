# -*- coding: utf-8 -*-
import os 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference


BATCH_SIZE = 100           #批数据100

LEARNING_RATE_BASE = 0.01  #基础学习率

LEARNING_RATE_DECAY = 0.99 #学习率的衰减率

REGUILARIZATION_RATE = 0.0001 #正则化系数

TRAINING_STEPS = 30000

MOVING_AVERAGE_DECAY =0.99 #滑动平均衰减率

MODEL_SAVE_PATH = "./mnist_cnn/model"
MODEL_NAME="cnn-mnist.ckpt"

def train(mnist):
    
    x=tf.placeholder(
        tf.float32,
        [BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
        name='x-input'
    )

    y_=tf.placeholder(
        tf.float32,
        [None,mnist_inference.OUTPUT_NODE],
        name="y-input"
    )

    regularizer=tf.contrib.layers.l2_regularizer(REGUILARIZATION_RATE)
    
    y=mnist_inference.inference(x,regularizer=regularizer,train=True)

    global_step=tf.Variable(0,trainable=False)

    variable_averages= tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    Variable_averages_op=variable_averages.apply(
        tf.trainable_variables()
    )

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,
        labels=tf.argmax(y_,1)
    )

    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    loss =cross_entropy_mean+tf.add_n(tf.get_collection("losses"))

    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    
    train_step=tf.train.GradientDescentOptimizer(
        learning_rate).minimize(
            loss,
            global_step=global_step
        )

    with tf.control_dependencies([train_step,Variable_averages_op]):
        train_op=tf.no_op(name="train")
    
    saver=tf.train.Saver()
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range (TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(
                BATCH_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS
            ))
            _, loss_value, step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

            if i%1000==0:
                print("After %d training step(s),validation accuracy "
                    "using average model is %g"%(step,loss_value))

        saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                    global_step=global_step
                )
        print("train finish")
        sess.close()

def main(argv=None):
    mnist =input_data.read_data_sets(
    "/path/to/MNIST_data",one_hot=True)
    
    train(mnist)

if __name__ == "__main__":
    main()
