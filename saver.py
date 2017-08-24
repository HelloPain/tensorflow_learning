import tensorflow as tf
import numpy as np
# W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32)
# b=tf.Variable([[1,2,3]],dtype=tf.float32)
#
# init=tf.global_variables_initializer()
#
# saver=tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path=saver.save(sess,"my_net/save.ckpt")
#     print("save to:",save_path)

W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32)
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32)

#不用initial

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save.ckpt")
    print("weights:",sess.run(W))
    print("biases:", sess.run(b))