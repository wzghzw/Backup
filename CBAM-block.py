import tensorflow as tf
#[batchsize, height, width, channel]NHWC
#TODO 权重必须与偏置初始化
def CBAM_func(x, channels, ratio=16, name='cbam',s_a_kernel_size=7) :
    with tf.compat.v1.variable_scope(name):
        with tf.compat.v1.variable_scope('channel_attention') :
            weights_initializer_c = tf.initializers.he_uniform()
            
            x_gap = tf.reduce_mean(x,axis=[1,2],keepdims=True)#global_avg_pooling(x)
            #相当于把图压缩了变成一个像素
            x_gap = tf.compat.v1.layers.dense(x_gap,units=channels//ratio,kernel_initializer=weights_initializer_c,activation=tf.nn.relu,name='fc1') #fully_connected(x_gap, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
            x_gap = tf.compat.v1.layers.dense(x_gap,units=channels,kernel_initializer=weights_initializer_c,name='fc2')#fully_connected(x_gap, units=channels, use_bias=use_bias, sn=sn, scope='fc2')


            x_gmp = tf.reduce_max(x,axis=[1,2],keepdims=True)
            x_gap = tf.compat.v1.layers.dense(x_gap,units=channels//ratio,bias_initializer=None,activation=tf.nn.relu,name='fc1',reuse=True) #fully_connected(x_gap, units=channels // ratio, use_bias=use_bias, sn=sn, scope='fc1')
            x_gap = tf.compat.v1.layers.dense(x_gap,units=channels,bias_initializer=None,name='fc2',reuse=True)
            
            scale = x_gap + x_gmp#[batchsize,1,1,channel]
            scale = tf.sigmoid(scale)#

            x = x * scale

        with tf.compat.v1.variable_scope('spatial_attention'):
            x_channel_avg_pooling = tf.reduce_mean(x, axis=-1, keepdims=True)#NHWC
            x_channel_max_pooling = tf.reduce_max(x, axis=-1, keepdims=True)
            scale = tf.concat([x_channel_avg_pooling, x_channel_max_pooling], axis=3)#channel concat

            
            weights_initializer = tf.initializers.he_uniform()
            tf.compat.v1.layers.conv2d(scale,filters=1,kernel_size=s_a_kernel_size,strides=1,padding="same",activation=None,use_bias=False,kernel_initializer=weights_initializer,name='conv')
            scale = tf.sigmoid(scale)

            x = x * scale

            return x


            '''
                    
            Set `reuse` to `True` when you only want to reuse existing Variables.
            Set `reuse` to `False` when you only want to create new Variables.
            Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want

            ValueError: when creating a new variable and shape is not declared,
            when reusing a variable and specifying a conflicting shape,
            or when violating reuse during variable creation.
            '''
