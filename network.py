

from tensorflow.python.keras import optimizers, regularizers
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import multi_gpu_model, Sequence
from tensorflow.python.keras import backend as K

from utils import ModelCache, DataProcessor

import tensorflow as tf
import numpy as np
import random

AUDIO_TO_VIDEO_RATIO = 4
SPEECH_ENTRY_IN_SEC = 10

class SpeechEnhancementNetwork(object):

    def __init__(self, spec_shape=None, vid_shape=None, num_filters=None, kernel_size=None, num_layers=None, model_cache_dir=None,
                 num_gpus=None, model=None, fit_model=None):
        self.__discriminator = load_model('/cs/labs/peleg/asaph/playground/avse/cache/models/classifier/disc.h5py')
        self.gpus = num_gpus
        self.model_cache = ModelCache(model_cache_dir)
        self.__model = model
        self.__fit_model = fit_model
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.spec_shape = spec_shape
        self.vid_shape = vid_shape
        self.num_layers = num_layers

    def  __build_AV_res_block(self, prev_vid, prev_audio, prev_delta, input_spec, pool):

        # vid = self.__distributed_2D_conv_block(prev_vid, pool)
        vid = prev_vid

        # tiled_vid = TimeDistributed(Flatten())(vid)
        tiled_vid = UpSampling1D(AUDIO_TO_VIDEO_RATIO)(vid)

        x = Concatenate()([tiled_vid, prev_audio, prev_delta, input_spec])
        res_delta = self.__conv_block(x, self.num_filters, self.kernel_size)
        res_delta = Conv1D(self.num_filters, self.kernel_size, padding='same')(res_delta)
        delta = Add()([res_delta, prev_delta])

        audio = Add()([self.__conv_block(prev_audio, self.num_filters, self.kernel_size), prev_audio])

        return vid, audio, delta

    @staticmethod
    def __conv_block(prev_x, num_filters, kernel_size, pool=0):
        x = Conv1D(num_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=regularizers.l2(),
                   bias_regularizer=regularizers.l2())(prev_x)
        x = LeakyReLU()(x)
        if pool:
            x = MaxPool1D(padding='same', strides=pool)(x)
        x = Dropout(0.5)(x)

        return x


    def build_discriminator(self, in_shape):
        input_spec = Input(in_shape)

        x = self.__conv_block(input_spec, 80, 5, pool=2)
        x = self.__conv_block(x, 80, 5, pool=2)
        x = self.__conv_block(x, 160, 5, pool=2)
        x = self.__conv_block(x, 160, 5)
        x = self.__conv_block(x, 320, 5)
        x = self.__conv_block(x, 320, 5)

        x = TimeDistributed(Flatten())(x)

        x = TimeDistributed(Dense(160, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2()))(x)
        x = TimeDistributed(LeakyReLU())(x)
        x = TimeDistributed(Dropout(0.5))(x)


        x = TimeDistributed(Dense(1, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2()))(x)
        x = GlobalAveragePooling1D()(x)

        out = Activation('sigmoid')(x)

        if self.gpus > 1:
            with tf.device('/cpu:0'):
                model = Model(inputs=[input_spec], outputs=[out], name='Discriminator')
                fit_model = multi_gpu_model(model, gpus=self.gpus)
        else:
            model = Model(inputs=[input_spec], outputs=[out], name='Discriminator')
            fit_model = model

        optimizer = optimizers.Adam(lr=1e-4)
        fit_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print 'Discriminator'
        model.summary(line_length=150)

        self.__discriminator = model
        self.__fit_discriminator = fit_model

    def train(self, train_speech_entries, train_noise_files, val_speech_entries, val_noise_files, batch_size):
        dp = DataProcessor(25, 16000, slice_len_in_ms=1000, split_to_batch=False)

        print 'num gpus: ', self.gpus

        print 'training discriminator'
        model = 'discriminator'
        train_disc_generator = DataGenerator(train_speech_entries,
                                             train_noise_files,
                                             dp,
                                             shuffle_noise=True,
                                             batch_size=batch_size,
                                             num_gpu=self.gpus,
                                             mode='discriminator')

        val_disc_generator = DataGenerator(val_speech_entries,
                                           val_noise_files,
                                           dp,
                                           shuffle_noise=True,
                                           batch_size=batch_size,
                                           num_gpu=self.gpus,
                                           mode='discriminator')

        SaveModel = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_disc())
        lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=1)
        self.__fit_discriminator.fit_generator(train_disc_generator,
                                               epochs=1000,
                                               callbacks=[SaveModel, lr_decay, early_stopping],
                                               validation_data=val_disc_generator,
                                               validation_steps=100,
                                               use_multiprocessing=True,
                                               max_queue_size=200,
                                               workers=4,
                                               verbose=1)

    def predict(self, mixed_spectrograms, video_samples):
        speech_spectrograms = self.__model.predict([video_samples, mixed_spectrograms], batch_size=1)[0]

        return np.squeeze(speech_spectrograms)

    def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):

        loss = self.__model.evaluate(x=[video_samples, mixed_spectrograms], y=speech_spectrograms, batch_size=1)

        return loss

    def save_disc(self):
        try:
            self.__discriminator.save(self.model_cache.disc_path())
        except Exception as e:
            print(e)

    @staticmethod
    def load(model_cache_dir):
        model_cache = ModelCache(model_cache_dir)
        model = load_model(model_cache.model_path(), custom_objects={'tf':K})

        return SpeechEnhancementNetwork(model=model)


class DataGenerator(Sequence):

    def __init__(self, speech_entries, noise_file_paths, data_processor, shuffle_noise=False, batch_size=4, num_gpu=1, mode='generator',
                 verbose=0):
        self.speech_entries = speech_entries
        self.noise_file_paths = noise_file_paths
        self.dp = data_processor
        self.shuffle_noise = shuffle_noise
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.noise_index = 0
        self.speech_index = 0
        self.cache = []
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        # number of batches (of size BATCH_SIZE) in the dataset
        return len(self.speech_entries) * 2

    def __getitem__(self, index):
        try:
            source_spectrogram, mixed_spectrogram = \
                self.dp.preprocess_sample2(self.speech_entries[index % len(self.speech_entries)], random.choice(self.noise_file_paths))

            clean = random.random() < 0.5

            if clean:
                specs = source_spectrogram.T
            else:
                specs = mixed_spectrogram.T

            specs = specs[:-(specs.shape[0] % self.dp.spec_frames_per_slice)]
            channels = specs.shape[1]
            specs = specs.reshape(-1, self.dp.spec_frames_per_slice, channels)

            if clean:
                labels = np.ones([specs.shape[0], 1])
            else:
                labels = np.zeros([specs.shape[0], 1])

            return specs, labels

        except Exception as e:
            pass

    def on_epoch_end(self):
        pass


if __name__ == '__main__':
    net = SpeechEnhancementNetwork(vid_shape=(None, 128, 128),
                                   spec_shape=(None, 80),
                                   num_filters=160,
                                   num_layers=3,
                                   kernel_size=5,
                                   num_gpus=1,
                                   model_cache_dir=None)

    # net.build()
    net.build_discriminator((None, 80))