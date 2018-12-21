import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
from threading import Thread, Lock
from queue import Queue
import gc
import tensorflow as tf


def load_train(train_path, classes):
    print('Going to read training images')
    images = []
    labels = []
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read files in {} (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)

        # Only load filenames and label indices
        for f in files:
            images.append(f)
            labels.append(index)

    return images, labels


class DataYielder:
    def __init__(self, train_path, image_size, classes, filenames, labels, max_size=20000, name='Train'):
        """
        This class is not thread safe

        Args:
            train_path:
            image_size:
            classes:
            validation_size:
            max_size:
        """
        self._image_size = image_size
        self._classes = classes
        self.train_path = train_path
        self._max_size = max_size
        self.name = name

        self._filenames = filenames
        self._label_indices = labels

        # Shape is amount of images, color count (RGB) and height and width
        self._images = np.zeros((len(labels), image_size, image_size, 3))
        self._labels = np.zeros((len(filenames), len(classes)))
        self._array_lock = Lock()
        self._load_images(0, max_size)

        self._image_request = Queue()
        self._running = True
        self._thread = Thread(target=self._update_loop)
        self._thread.start()

    def stop(self):
        self._running = False
        self._image_request.put(None)
        self._thread.join()

    def _update_loop(self):
        while self._running:
            data = self._image_request.get()
            if data is None:
                continue

            if self.max_size > self._images.shape[0]:
                continue

            start, stop = data
            length = stop-start
            if start + self.max_size > self._images.shape[0]:
                start_beginning = start + self.max_size - self._images.shape[0]
                self._load_images(0, start_beginning)

                if start_beginning != length:
                    self._load_images(self._images.shape[0]-length+start_beginning, self._images.shape[0])

            else:
                self._load_images(start+self.max_size, stop+self.max_size)

            stop = start
            start -= length
            # In case we try to unload data with negative indices we need to
            # make sure stop is None to correctly slice starting from the back
            if start-length < 0:
                stop = None
            self._unload_images(start-length, stop)

    def _unload_images(self, start, stop):
        if start is None or stop is None:
            return

        #print(f'[{self.name}] Unloading {start}-{stop}')

        with self._array_lock:
            self._images[start:stop] = 0
            self._labels[start:stop] = 0

    def _load_images(self, start, stop):
        stop = min(self._images.shape[0], stop)
        #print(f'[{self.name}] Loading images in the range {start}-{stop}')
        for i in range(start, stop):
            f = self._filenames[i]
            image = cv2.imread(f)
            image = cv2.resize(image, (self._image_size, self._image_size), 0, 0,
                               cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            label = np.zeros(len(self._classes))
            index = self._label_indices[i]
            label[index] = 1.0

            with self._array_lock:
                self._images[i] = image
                self._labels[i] = label

        return True

    @property
    def max_size(self):
        return self._max_size

    def __getitem__(self, item):
        if isinstance(item, slice):
            amount = item.stop-item.start
            if amount > self.max_size:
                raise ValueError(f'Max size is smaller than batch size {amount} > {self.max_size}')

            self._image_request.put((item.start, item.stop))
            with self._array_lock:
                return self._images[item], self._labels[item]

    def __len__(self):
        return self._images.shape[0]


class DataSet(object):

    def __init__(self, images):
        self._num_examples = len(images)

        self._images = images
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end]


def read_train_sets(train_path, image_size, classes, validation_size, max_size=5000):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels = load_train(train_path, classes)
    images, labels = shuffle(images, labels)

    old_val = validation_size

    if isinstance(validation_size, float):
        validation_size = int(validation_size * len(images))

    validation_files = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_files = images[validation_size:]
    train_labels = labels[validation_size:]

    def read_image(filename, label):
        image = cv2.imread(filename.decode('utf-8'))
        image = cv2.resize(image, (image_size, image_size), 0, 0,
                           cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = image[:, :, (2, 1, 0)]  # BGR to RGB
        image = np.multiply(image, 1.0 / 255.0)

        label_ = np.zeros(len(classes))
        label_[label] = 1.0

        return image, label_

    train_set = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_set = train_set.map(lambda filename, label: tuple(tf.py_func(read_image, [filename, label], [tf.float32, tf.float64])))
    train_set = train_set.prefetch(max_size)

    validation_set = tf.data.Dataset.from_tensor_slices((validation_files, validation_labels))
    validation_set = validation_set.map(lambda filename, label: tuple(tf.py_func(read_image, [filename, label], [tf.float32, tf.float64])))
    validation_set = validation_set.prefetch(int(max_size*old_val))
    print("Completed reading of input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t{}".format(len(train_labels)))
    print("Number of files in Validation-set:\t{}".format(len(validation_labels)))

    return train_set, validation_set, len(train_files)
