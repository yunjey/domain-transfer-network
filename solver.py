import tensorflow as tf
import numpy as np
import os
import scipy.io 
import hickle
from scipy import ndimage


class Solver(object):
    """Load dataset and train DCGAN"""
    
    def __init__(self, model, num_epoch=10, mnist_path= 'mnist/', svhn_path='svhn/', model_save_path='model/', log_path='log/'):
        self.model = model
        self.num_epoch = num_epoch
        self.mnist_path = mnist_path
        self.svhn_path = svhn_path
        self.model_save_path = model_save_path
        self.log_path = log_path
        
        # create directory if not exists
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        # construct the dcgan model
        model.build_model()
        
        # load dataset
        self.svhn = self.load_svhn(self.svhn_path)
        self.mnist = self.load_mnist(self.mnist_path)
        
        
    def load_svhn(self, image_path, split='train'):
        print ('loading svhn image dataset..')
        if split == 'train':
            svhn = scipy.io.loadmat(os.path.join(image_path, 'train_32x32.mat'))
        else:
            svhn = scipy.io.loadmat(os.path.join(image_path, 'test_32x32.mat'))
            
        images = np.transpose(svhn['X'], [3, 0, 1, 2])    
        images = images / 127.5 - 1
        print ('finished loading svhn image dataset..!')
        return images
    
    
    def load_mnist(self, image_path, split='train'):
        print ('loading mnist image dataset..')
        if split == 'train':
            image_file = os.path.join(image_path, 'train.images.hkl')
        else:
            image_file = os.path.join(image_path, 'test.images.hkl')
        
        try:
            images = hickle.load(image_file)
        except:
            hickle.load(images, image_file)
            
        images = images / 127.5 - 1
        print ('finished loading mnist image dataset..!')
        return images
    
    
    def train(self):
        model=self.model
        
        #load image dataset
        svhn = self.svhn
        mnist = self.mnist
        
        num_iter_per_epoch = int(mnist.shape[0] / model.batch_size)
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # initialize parameters
            tf.initialize_all_variables().run()
            summary_writer = tf.train.SummaryWriter(logdir=self.log_path, graph=tf.get_default_graph())
             
            for e in range(self.num_epoch):
                for i in range(num_iter_per_epoch):
                    
                    # train model for domain S
                    image_batch = svhn[i*model.batch_size:(i+1)*model.batch_size]
                    feed_dict = {model.images: image_batch}
                    sess.run(model.d_optimizer_fake, feed_dict)
                    sess.run(model.f_optimizer_const, feed_dict)
                    sess.run(model.g_optimizer, feed_dict)
                    
                    if i % 10 == 0:
                        feed_dict = {model.images: image_batch}
                        summary, d_loss, g_loss = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
                        summary_writer.add_summary(summary, e*num_iter_per_epoch + i)
                        print ('Epoch: [%d] Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' %(e+1, i+1, num_iter_per_epoch, d_loss, g_loss))
                    
                    # train model for domain T
                    image_batch = mnist[i*model.batch_size:(i+1)*model.batch_size]
                    feed_dict = {model.images: image_batch}
                    sess.run(model.d_optimizer_real, feed_dict)
                    sess.run(model.d_optimizer_fake, feed_dict)
                    sess.run(model.g_optimizer, feed_dict)
                    sess.run(model.g_optimizer_const, feed_dict)
                    
                 
                        
                    if i % 500 == 0:  
                        model.saver.save(sess, os.path.join(self.model_save_path, 'dcgan-%d' %(e+1)), global_step=i+1) 
                        print ('model/dcgan-%d-%d saved' %(e+1, i+1))