

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
        self.gpus = num_gpus
        self.model_cache = ModelCache(model_cache_dir)
        self.__model = model
        self.__fit_model = fit_model
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.spec_shape = spec_shape
        self.vid_shape = vid_shape
        self.num_layers = num_layers

    def __build_AV_res_block(self, prev_vid, prev_audio, prev_delta, input_spec, pool):

        # vid = self.__distributed_2D_conv_block(prev_vid, pool)
        vid = prev_vid

        # tiled_vid = TimeDistributed(Flatten())(vid)
        tiled_vid = UpSampling1D(AUDIO_TO_VIDEO_RATIO)(vid)

        x = Concatenate()([tiled_vid, prev_audio, prev_delta, input_spec])
        res_delta = self.__conv_block(x, self.num_filters, self.kernel_size)
        res_delta = Conv1D(self.num_filters, self.kernel_size, padding='same')(res_delta)
        delta = Add()([res_delta, prev_delta])

        audio = Add()([self.__conv_block(prev_audio, self.num_filters, 5), prev_audio])

        return vid, audio, delta

    @staticmethod
    def __conv_block(prev_x, num_filters, kernel_size):
        x = Conv1D(num_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=regularizers.l2(),
                   bias_regularizer=regularizers.l2())(prev_x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        # if pool:
        #     x = MaxPool1D(padding='same', strides=pool)(x)
        x = Dropout(0.5)(x)

        return x

    @staticmethod
    def __distributed_2D_conv_block(prev_x, pool, num_filters, kernel_size):
        x = TimeDistributed(Conv2D(num_filters,
                                   (kernel_size, kernel_size),
                                   padding='same',
                                   kernel_regularizer=regularizers.l2(),
                                   bias_regularizer=regularizers.l2()))(prev_x)
        x = BatchNormalization()(x)
        x = TimeDistributed(LeakyReLU())(x)
        if pool:
            x = TimeDistributed(MaxPool2D(strides=(pool, pool), padding='same'))(x)

        x = TimeDistributed(Dropout(0.5))(x)

        return x


    # def build_discriminator(self, in_shape):
    #     input_spec = Input(in_shape)
    #
    #     x = self.__conv_block(input_spec, 80, 5, pool=2)
    #     x = self.__conv_block(x, 80, 5, pool=2)
    #     x = self.__conv_block(x, 160, 5, pool=2)
    #     x = self.__conv_block(x, 160, 5)
    #     x = self.__conv_block(x, 320, 5)
    #     x = self.__conv_block(x, 320, 5)
    #
    #     x = TimeDistributed(Flatten())(x)
    #
    #     x = TimeDistributed(Dense(160, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2()))(x)
    #     # x = TimeDistributed(BatchNormalization())(x)
    #     x = TimeDistributed(LeakyReLU())(x)
    #     x = TimeDistributed(Dropout(0.5))(x)
    #
    #     x = TimeDistributed(Dense(1))(x)
    #
    #     x = TimeDistributed(Activation('sigmoid'))(x)
    #     out = GlobalAveragePooling1D()(x)
    #
    #     if self.gpus > 1:
    #         with tf.device('/cpu:0'):
    #             model = Model(inputs=[input_spec], outputs=[out], name='Discriminator')
    #             fit_model = multi_gpu_model(model, gpus=self.gpus)
    #     else:
    #         model = Model(inputs=[input_spec], outputs=[out], name='Discriminator')
    #         fit_model = model
    #
    #     optimizer = optimizers.Adam(lr=1e-4)
    #     fit_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #
    #     print 'Discriminator'
    #     model.summary(line_length=150)
    #
    #     self.__discriminator = model
    #     self.__fit_discriminator = fit_model


    def build(self):
        input_spec = Input(self.spec_shape)
        input_vid = Input(self.vid_shape)

        # video encoder
        vid = Lambda(lambda a: K.expand_dims(a, -1))(input_vid)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=3)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=3)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=5)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=5)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=5)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=5)
        vid = self.__distributed_2D_conv_block(vid, pool=2, num_filters=80, kernel_size=5)

        vid = TimeDistributed(Flatten())(vid)

        vid = Conv1D(80, 5, padding='same', kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(vid)
        vid = BatchNormalization()(vid)
        vid = TimeDistributed(LeakyReLU())(vid)

        vid = Conv1D(80, 5, padding='same', kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(vid)
        vid = BatchNormalization()(vid)
        vid = TimeDistributed(LeakyReLU())(vid)

        vid = Conv1D(80, 5, padding='same', kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(vid)
        vid = BatchNormalization()(vid)
        vid = TimeDistributed(LeakyReLU())(vid)

        vid = Conv1D(80, 5, padding='same', kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(vid)

        tiled_vid = UpSampling1D(AUDIO_TO_VIDEO_RATIO)(vid)

        # first audio conv - not res
        audio = self.__conv_block(input_spec, self.num_filters, self.kernel_size)

        # first delta - not res
        x = Concatenate()([tiled_vid, audio])
        delta = self.__conv_block(x, self.num_filters, self.kernel_size)

        # res blocks
        for i in range(self.num_layers):
            vid, audio, delta = self.__build_AV_res_block(vid, audio, delta, input_spec, pool=0)

        tiled_vid = TimeDistributed(Flatten())(vid)
        tiled_vid = UpSampling1D(AUDIO_TO_VIDEO_RATIO)(tiled_vid)

        x = Concatenate()([tiled_vid, audio, delta, input_spec])
        # delta = self.__conv_block(x, 80, 5)
        delta = TimeDistributed(Dense(256))(x)
        delta = BatchNormalization()(delta)
        delta = TimeDistributed(LeakyReLU())(delta)

        delta = TimeDistributed(Dense(256))(delta)
        delta = BatchNormalization()(delta)
        delta = TimeDistributed(LeakyReLU())(delta)

        delta = TimeDistributed(Dense(80))(delta)

        out = Add()([delta, input_spec])

        # todo: replace with 'classifier'
        self.__discriminator = load_model('/cs/labs/peleg/asaph/playground/avse/cache/models/classifier_cluster_no_batchnorm_0.5/disc.h5py')

        for layer in self.__discriminator.layers:
            layer.trainable = False

        disc_out = self.__discriminator(out)

        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        if self.gpus > 1:
            with tf.device('/cpu:0'):
                model = Model(inputs=[input_vid, input_spec], outputs=[out, disc_out], name='Net')
                fit_model = multi_gpu_model(model, gpus=self.gpus)
        else:
            model = Model(inputs=[input_vid, input_spec], outputs=[out, disc_out], name='Net')
            fit_model = model

        optimizer = optimizers.Adam(lr=5e-4)
        fit_model.compile(loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1.0, 0.2], optimizer=optimizer, options=run_opts)

        # print 'Net'
        # model.summary(line_length=150)

        self.__model = model
        self.__fit_model = fit_model



    def train(self, train_speech_entries, train_noise_files, val_speech_entries, val_noise_files, batch_size):
        dp = DataProcessor(25, 16000, slice_len_in_ms=1000, split_to_batch=True)

        print 'num gpus: ', self.gpus

        # print 'training discriminator'
        # model = 'discriminator'
        # train_disc_generator = DataGenerator(train_speech_entries,
        #                                      train_noise_files,
        #                                      dp,
        #                                      shuffle_noise=True,
        #                                      batch_size=batch_size,
        #                                      num_gpu=self.gpus,
        #                                      mode='discriminator')
        #
        # val_disc_generator = DataGenerator(val_speech_entries,
        #                                    val_noise_files,
        #                                    dp,
        #                                    shuffle_noise=True,
        #                                    batch_size=batch_size,
        #                                    num_gpu=self.gpus,
        #                                    mode='discriminator')
        #
        # SaveModel = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_model())
        # lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=1)
        # self.__fit_discriminator.fit_generator(train_disc_generator,
        #                                        epochs=1000,
        #                                        callbacks=[SaveModel, lr_decay, early_stopping],
        #                                        validation_data=val_disc_generator,
        #                                        validation_steps=100,
        #                                        use_multiprocessing=True,
        #                                        max_queue_size=200,
        #                                        workers=4,
        #                                        verbose=1)

        print 'training generator'
        train_adver_generator = DataGenerator(train_speech_entries,
                                              train_noise_files,
                                              dp,
                                              shuffle_noise=True,
                                              batch_size=batch_size,
                                              num_gpu=self.gpus)

        val_adver_generator = DataGenerator(val_speech_entries,
                                            val_noise_files,
                                            dp,
                                            shuffle_noise=True,
                                            batch_size=batch_size,
                                            num_gpu=self.gpus)

        SaveModel = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_model())
        lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1)
        self.__fit_model.fit_generator(train_adver_generator,
                                       epochs=1000,
                                       callbacks=[SaveModel, lr_decay, early_stopping],
                                       validation_data=val_adver_generator,
                                       validation_steps=100,
                                       use_multiprocessing=True,
                                       max_queue_size=20,
                                       workers=self.gpus,
                                       verbose=1)


    def predict(self, mixed_spectrograms, video_samples):
        speech_spectrograms = self.__model.predict([video_samples, mixed_spectrograms], batch_size=1)

        return np.squeeze(speech_spectrograms)

    def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):

        loss = self.__model.evaluate(x=[video_samples, mixed_spectrograms], y=speech_spectrograms, batch_size=1)

        return loss

    def save_model(self):
        try:
            self.__model.save(self.model_cache.model_path())
            self.__model.save(self.model_cache.model_backup_path())
        except Exception as e:
            print(e)

    @staticmethod
    def load(model_cache_dir):
        model_cache = ModelCache(model_cache_dir)
        model = load_model(model_cache.model_path(), custom_objects={'tf':K})

        return SpeechEnhancementNetwork(model=model)


# class DataGenerator(Sequence):
#
#     def __init__(self, speech_entries, noise_file_paths, data_processor, shuffle_noise=False, batch_size=4, num_gpu=1, mode='generator',
#                  verbose=0):
#         self.speech_entries = speech_entries
#         self.noise_file_paths = noise_file_paths
#         self.dp = data_processor
#         self.shuffle_noise = shuffle_noise
#         self.batch_size = batch_size
#         self.num_gpu = num_gpu
#         self.noise_index = 0
#         self.speech_index = 0
#         self.cache = []
#         self.mode = mode
#         self.verbose = verbose
#
#     def __len__(self):
#         # number of batches (of size BATCH_SIZE) in the dataset
#         return len(self.speech_entries) * 2
#
#     def __getitem__(self, index):
#         try:
#             source_spectrogram, mixed_spectrogram = \
#                 self.dp.preprocess_sample2(self.speech_entries[index % len(self.speech_entries)], random.choice(self.noise_file_paths))
#
#             clean = random.random() < 0.5
#
#             if clean:
#                 specs = source_spectrogram.T
#             else:
#                 specs = mixed_spectrogram.T
#
#             specs = specs[:-(specs.shape[0] % self.dp.spec_frames_per_slice)]
#             channels = specs.shape[1]
#             specs = specs.reshape(-1, self.dp.spec_frames_per_slice, channels)
#
#             if clean:
#                 labels = np.ones([specs.shape[0], 1])
#             else:
#                 labels = np.zeros([specs.shape[0], 1])
#
#             return specs, labels
#
#         except Exception as e:
#             pass
#
#     def on_epoch_end(self):
#         pass

class DataGenerator(Sequence):

    def __init__(self, speech_entries, noise_file_paths, data_processor, shuffle_noise=False, batch_size=4, num_gpu=1):
        self.speech_entries = speech_entries
        self.noise_file_paths = noise_file_paths
        self.dp = data_processor
        self.shuffle_noise = shuffle_noise
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.noise_index = 0
        self.speech_index = 0

    def __len__(self):
        return int(len(self.speech_entries) * 10000 / self.dp.slice_len_in_ms / self.batch_size / self.num_gpu)

    def __getitem__(self, index):
        try:
            video_samples, mixed_spectrograms, source_spectrograms = self.dp.preprocess_sample(
                self.speech_entries[index % len(self.speech_entries)],
                random.choice(self.noise_file_paths))
            ind = random.sample(range(video_samples.shape[0]), self.batch_size * self.num_gpu)
            return [video_samples[ind], mixed_spectrograms[ind]], [source_spectrograms[ind], np.ones([len(ind), 1])]
        except Exception as e:
            print e
            print 'failed processing'
            pass

    def on_epoch_end(self):
        pass
# class DataGenerator(Sequence):
#
#     def __init__(self, speech_entries, noise_file_paths, data_processor, shuffle_noise=False, batch_size=4, num_gpu=1, mode='generator', verbose=0):
#         self.speech_entries = speech_entries
#         self.noise_file_paths = noise_file_paths
#         self.dp = data_processor
#         self.shuffle_noise = shuffle_noise
#         self.batch_size = min(batch_size, SPEECH_ENTRY_IN_SEC * 1000 / self.dp.slice_len_in_ms) # todo: revert to batch_size after improving __get
#         self.num_gpu = num_gpu
#         self.noise_index = 0
#         self.speech_index = 0
#         self.cache = []
#         self.mode = mode
#         self.verbose = verbose
#
#     def __len__(self):
#         # number of batches (of size BATCH_SIZE) in the dataset
#         return len(self.speech_entries) * SPEECH_ENTRY_IN_SEC * 1000 / self.dp.slice_len_in_ms / (self.batch_size * self.num_gpu) * 2
#
#     def __getitem__(self, index):
#         if self.verbose:
#             print 'index', index
#             print 'speech', self.speech_index
#         if len(self.cache) != 0:
#             tup = self.cache.pop()
#             if self.verbose:
#                 print tup[1][0].shape, tup[1][0][0]
#             return tup
#
#         if self.speech_index >= len(self.speech_entries):
#             self.speech_index = 0
#         if self.noise_index >= len(self.noise_file_paths):
#             self.noise_index = 0
#             if self.shuffle_noise:
#                 random.shuffle(self.noise_file_paths)
#
#         try:
#             video_samples, mixed_spectrograms, mixed_phases, source_spectrograms, source_phases = \
#                 self.dp.preprocess_sample(self.speech_entries[self.speech_index], self.noise_file_paths[self.noise_index])
#
#             self.noise_index += 1
#             self.speech_index += 1
#
#             raw_batch_size = video_samples.shape[0]
#             for j in range(0, raw_batch_size, self.batch_size * self.num_gpu):
#                 vid = video_samples[j : j + (self.batch_size * self.num_gpu)]
#                 mix = mixed_spectrograms[j : j + (self.batch_size * self.num_gpu)]
#                 source = source_spectrograms[j : j + (self.batch_size * self.num_gpu)]
#
#                 if vid.shape[0] == 0 or vid.shape[0] < self.num_gpu:
#                     continue
#
#                 if self.mode == 'adversarial':
#                     self.cache.append(([vid, mix], [source, np.random.rand(vid.shape[0], 1) * 0.1 + 0.9]))
#                 else:
#                     self.cache.append(([source], [np.ones([mix.shape[0], 1])]))
#                     self.cache.append(([mix], [np.zeros([mix.shape[0], 1])]))
#
#
#             tup = self.cache.pop()
#             if self.verbose:
#                 print tup[1][0].shape, tup[1][0][0]
#             return tup
#         except Exception as e:
#             pass
#
#     def on_epoch_end(self):
#         pass


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