import tensorflow as tf
import os
import numpy as np
import pdb

#NUM_SUPER_CLASSES = 20
#NUM_CLASSES = 100
#NUM_TRAIN = 50000
#NUM_TEST = 10000

HEIGHT = 84
WIDTH = 84
DEPTH = 3

#LABEL_BYTES = 2

class TieredImageNet(object):
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

        self.height = HEIGHT
        self.width = WIDTH
        self.depth = DEPTH

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        filenames = os.listdir(self.data_dir)
        self.labels = [int(fn.split('_')[0].replace('C','')) for fn in filenames]
        self.filenames = [os.path.join(self.data_dir, fn) for fn in filenames]

        self.n_classes = len(np.unique(self.labels))
        self.n_images = len(self.labels)

        path = tf.constant(self.filenames)
        label = tf.constant(self.labels)

        def preprocess(path, label):
            img_str = tf.read_file(path)
            img_decoded = tf.image.decode_jpeg(img_str, channels=self.depth)
            img_std = self.std_with_knwon_moments(img_decoded)
            return img_std, tf.one_hot(label, self.n_classes)

        dataset = tf.data.Dataset.from_tensor_slices((path, label))
        dataset = dataset.map(preprocess)

        return dataset

    def std_with_knwon_moments(self, image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        mu = tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1,1,3])
        sig = tf.reshape(tf.constant([0.229, 0.224, 0.225]),[1,1,3])
        image = (image - mu) / (sig + 1e-8)
        return image


    def augment(self, image, label):
        image = tf.random_crop(image, [self.height, self.width, self.depth])
        image = tf.image.random_flip_left_right(image)
        return image, label

    
    def input_fn(self, batch_size, train_mode):
        dataset = self.dataset
        if train_mode:
            dataset = dataset.map(self.augment, num_parallel_calls=20)
            dataset = dataset.shuffle(buffer_size=2000).repeat().batch(batch_size)
        else:
            dataset = dataset.repeat().batch(batch_size)
        
#        dataset = dataset.prefetch(8*batch_size)
#        dataset = dataset.repeat()
            
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        #images = tf.transpose(images, [0,3,1,2])
        return images, labels
        

if __name__=='__main__':
    dataset = TieredImageNet('tiered', 'images/val')
    x, y = dataset.input_fn(32, train_mode=True)
    #tx, ty = dataset.input_fn(40, train_mode=False)

    sess = tf.Session()
    px, py = sess.run([x,y])
    pdb.set_trace()
