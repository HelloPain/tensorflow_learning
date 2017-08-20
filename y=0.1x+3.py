import tensorflow as tf
import numpy as np
#create data
x_d=np.random.rand(100).astype(np.float32)
y_d=x_d*0.1+0.3
#create tensorflow structure start
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))#一维（-1,1）的随机数
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_d+biases

loss=tf.reduce_mean(tf.square(y-y_d))
optimizer=tf.train.GradientDescentOptimizer(0.5)#优化器
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
#finished

sess=tf.Session()
sess.run(init)
#激活初始化！！！

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))
