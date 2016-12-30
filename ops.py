import tensorflow as tf
    

class batch_norm(object):
    """Computes batch normalization operation
    
    Args:
        x: input tensor of shape (batch_size, width, height, channels_in) or (batch_size, dim_in)
        train: True or False;  At train mode, it normalizes the input with mini-batch statistics
                               At test mode, it normalizes the input with the moving averages and variances
 
    Returns:
        out: batch normalized output of the same shape with x
    """
    def __init__(self, name):
        self.name = name
    
    def __call__(self, x, train=True):
        out = tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True, activation_fn=None, 
                                           updates_collections=None, is_training=train, scope=self.name)
        return out
    
    
def conv2d(x, channel_out, k_w=5, k_h=5, s_w=2, s_h=2, name=None):
    """Computes convolution operation
    
    Args:
        x: input tensor of shape (batch_size, width_in, heigth_in, channel_in)
        channel_out: number of channel for output tensor
        k_w: kernel width size; default is 5
        k_h: kernel height size; default is 5
        s_w: stride size for width; default is 2
        s_h: stride size for heigth; default is 2
        
    Returns:
        out: output tensor of shape (batch_size, width_out, height_out, channel_out)
    """
    channel_in = x.get_shape()[-1]
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_w, k_h, channel_in, channel_out], 
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[channel_out], initializer=tf.constant_initializer(0.0))
        
        out = tf.nn.conv2d(x, w, strides=[1, s_w, s_h, 1], padding='SAME') + b
        
        return out
    
    
def deconv2d(x, output_shape, k_w=5, k_h=5, s_w=2, s_h=2, name=None):
    """Computes deconvolution operation
    
    Args:
        x: input tensor of shape (batch_size, width_in, height_in, channel_in)
        output_shape: list corresponding to [batch_size, width_out, height_out, channel_out]
        k_w: kernel width size; default is 5
        k_h: kernel height size; default is 5
        s_w: stride size for width; default is 2
        s_h: stride size for heigth; default is 2
        
    Returns:
        out: output tensor of shape (batch_size, width_out, hegith_out, channel_out)
    """
    channel_in = x.get_shape()[-1]
    channel_out = output_shape[-1]
    
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_w, k_h, channel_out, channel_in], 
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[channel_out], initializer=tf.constant_initializer(0.0))
        
        out = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, s_w, s_h, 1]) + b
    
        return out
    
def linear(x, dim_out, name=None):
    """Computes linear transform (fully-connected layer)
    
    Args:
        x: input tensor of shape (batch_size, dim_in)
        dim_out: dimension for output tensor
        
    Returns:
        out: output tensor of shape (batch_size, dim_out)
    """
    dim_in = x.get_shape()[-1]
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[dim_in, dim_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[dim_out], initializer=tf.constant_initializer(0.0))
        
        out = tf.matmul(x, w) + b
        
        return out
    

def relu(x):
    return tf.nn.relu(x)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)