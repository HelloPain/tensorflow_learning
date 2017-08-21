import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

x_d=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_d.shape)
y_d=np.square(x_d)+noise-0.05

sx=tf.placeholder(tf.float32,[None,1])
sy=tf.placeholder(tf.float32,[None,1])

l1=add_layer(sx,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(sy-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_d,y_d)#用点画出x_d,y_d
plt.ion()#一直画出
plt.show()

for i in range(1000):
   sess.run(train_step,feed_dict={sx: x_d, sy: y_d})
   if(i%50==0):
       #print(sess.run(loss,feed_dict={sx: x_d, sy: y_d}))
       try:
           ax.lines.remove(lines[0])
       except Exception:
            pass
       prediction_va=sess.run(prediction,feed_dict={sx:x_d})
       lines=ax.plot(x_d,prediction_va,'r-',lw=5)#用线画
       plt.pause(0.1)
plt.pause(10)