import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def resize_images(image_arrays, size=[32, 32]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)
        
        resized_image_arrays[i] = np.asarray(resized_image)
    
    return np.expand_dims(resized_image_arrays, 3)  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    mnist = input_data.read_data_sets(train_dir='mnist')

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}
    
    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}
        
    save_pickle(train, 'mnist/train.pkl')
    save_pickle(test, 'mnist/test.pkl')
    
    
if __name__ == "__main__":
    main()
    