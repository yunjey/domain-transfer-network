import tensorflow as tf
from model import DTN
from solver import Solver



flags = tf.app.flags
flags.DEFINE_boolean('is_train', False, 'True if train mode, False if test mode')

FLAGS = flags.FLAGS

def main(_):
    
    model = DTN(batch_size=100, learning_rate=0.001, image_size=32, output_size=32, 
                 dim_color=3, dim_fout=100, dim_df=64, dim_gf=64, dim_ff=64)
    
    solver = Solver(model, num_epoch=10, mnist_path= 'mnist/', svhn_path='svhn/', model_save_path='model/', 
                    log_path='log/', sample_path='sample/', test_model_path='model/dtn-2-1', sample_iter=100)
    
    
    if FLAGS.is_train:
        solver.train()
    else:
        solver.test()
    


if __name__ == '__main__':
    tf.app.run()
    
    
   