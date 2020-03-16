
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

channels = 64
ratio = 16
x = tf.random_normal(shape=[10, 32, 32, 64])
with tf.variable_scope('channel_attention',reuse=tf.AUTO_REUSE) :
    weights_initializer_c = tf.initializers.he_uniform()
    
    x_gap = tf.reduce_mean(x,axis=[1,2],keepdims=True)#global_avg_pooling(x)
    #相当于把图压缩了变成一个像素
    x_gap = tf.layers.dense(x_gap,units=channels//ratio,kernel_initializer=weights_initializer_c,use_bias=False,activation=tf.nn.relu,name='fc1') #fully_connected(x_gap, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
    x_gap = tf.layers.dense(x_gap,units=channels,kernel_initializer=weights_initializer_c,use_bias=False,name='fc2')#fully_connected(x_gap, units=channels, use_bias=use_bias, sn=sn, scope='fc2')


    x_gmp = tf.reduce_max(x,axis=[1,2],keepdims=True)
    x_gap = tf.layers.dense(x_gap,units=channels//ratio,bias_initializer=None,use_bias=False,activation=tf.nn.relu,name='fc1') #fully_connected(x_gap, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
    x_gap = tf.layers.dense(x_gap,units=channels,bias_initializer=None,use_bias=False,name='fc2')
    



print([x.name for x in tf.global_variables()])