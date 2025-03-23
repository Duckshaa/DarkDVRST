import numpy as np
import tensorflow as tf
import os

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size=32, frame_size=(224, 224), num_frames=16, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_paths = [self.video_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(batch_video_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_video_paths, batch_labels):
        X = np.array([video_to_npy(video) for video in batch_video_paths])
        y = np.array(batch_labels)
        return X, y
