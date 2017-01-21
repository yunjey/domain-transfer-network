import tensorflow as tf
from model import DTN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
FLAGS = flags.FLAGS

def main(_):
    
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003)
    solver = Solver(model, batch_size=100, pretrain_iter=5000, train_iter=2000, sample_iter=100, 
                    svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', model_save_path='model')
    
    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()
        
if __name__ == '__main__':
    tf.app.run()