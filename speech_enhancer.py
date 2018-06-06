
import argparse, pickle, sys

from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from shutil import copy2
from mediaio import ffmpeg
from utils import *

# sys.path.append('/usr/local/jetbrains/pycharm-2017.2.2/debug-eggs/pycharm-debug.egg')
# import pydevd
#
# pydevd.settrace('cooper-01', port=12345, stdoutToServer=True, stderrToServer=True)

BASE_FOLDER = '/cs/labs/peleg/asaph/playground/avse' # todo: remove before releasing code
SPLIT = 10

def preprocess(args):
    assets = AssetManager(args.base_folder)
    assets.create_preprocessed_data_dir(args.data_name)

    dataset_path = assets.get_data_set_dir(args.dataset, group='test')

    speaker_ids = list_speakers(args.speakers, args.ignored_speakers, dataset_dir=dataset_path)

    speech_entries, noise_file_paths = list_data(
        dataset_path, speaker_ids, args.noise_dirs, max_files=args.number_of_samples
    )

    video_samples, mixed_spectrograms, mixed_phases, source_spectrograms, source_phases, source_waveforms, metadatas = preprocess_data(
        speech_entries, noise_file_paths, args.cpus)

    np.savez(
        assets.get_preprocessed_blob_data_path(args.data_name),
        video_samples=video_samples,
        mixed_spectrograms=mixed_spectrograms,
        mixed_phases=mixed_phases,
        source_spectrograms=source_spectrograms,
        source_phases=source_phases,
        source_waveforms=source_waveforms
    )

    with open(assets.get_preprocessed_blob_metadata_path(args.data_name), 'wb') as preprocessed_fd:
        pickle.dump(metadatas, preprocessed_fd)

def train(args):
    assets = AssetManager(args.base_folder)
    assets.create_model(args.model)

    train_dataset_path = assets.get_data_set_dir(args.dataset_dir)
    if args.val_dataset_dir:
        val_dataset_path = assets.get_data_set_dir(args.val_dataset_dir, group='val')
    else:
        val_dataset_path = assets.get_data_set_dir(args.dataset_dir, group='val')

    print 'listing train speakers...'
    train_speaker_ids = list_speakers(args.train_speakers, args.train_ignored_speakers, train_dataset_path)


    print 'listing train data...'
    train_speech_entries, train_noise_file_paths = list_data(
        train_dataset_path, train_speaker_ids, args.noise_dirs, max_files=args.number_of_samples, shuffle=True
    )

    print 'listing val speakers...'
    val_speaker_ids = list_speakers(args.val_speakers, args.val_ignored_speakers, val_dataset_path)

    print 'listing val data...'
    if not args.val_noise_dirs:
        val_noise_dirs = [noise_dir.replace('train', 'val') for noise_dir in args.noise_dirs]
    else:
        val_noise_dirs = args.val_noise_dirs

    if args.number_of_samples:
        val_number_of_samples = args.number_of_samples
    else:
        val_number_of_samples = None

    val_speech_entries, val_noise_file_paths = list_data(
        val_dataset_path, val_speaker_ids, val_noise_dirs, max_files=val_number_of_samples, shuffle=True
    )

    print 'train dataset: ', train_dataset_path
    print 'num train speakers: ', len(train_speaker_ids)
    print 'num train clean files: ', len(train_speech_entries)
    print 'num train noise files: ', len(train_noise_file_paths)

    print 'val dataset: ', val_dataset_path
    print 'num val speakers: ', len(val_speaker_ids)
    print 'num val clean files: ', len(val_speech_entries)
    print 'num val noise files: ', len(val_noise_file_paths)

    print 'building network...'
    network = SpeechEnhancementNetwork(spec_shape=(None, 80),
                                       vid_shape=(None, 128, 128),
                                       num_filters=160,
                                       kernel_size=5,
                                       num_layers=20,
                                       model_cache_dir=assets.get_model_cache_path(args.model),
                                       num_gpus=args.gpus)
    network.build()
    network.train(train_speech_entries, train_noise_file_paths, val_speech_entries, val_noise_file_paths)

    # network.save(model_cache_dir)


def predict(args):
    assets = AssetManager(args.base_folder)
    assets.create_prediction_dir(args.model)

    testset_path = assets.get_preprocessed_blob_data_path(args.data_name)
    metadata_path = assets.get_preprocessed_blob_metadata_path(args.data_name)

    vid, mix_specs, source_specs, source_phases, mixed_phases, source_waveforms = load_preprocessed_samples(testset_path,
                                                                                                      max_samples=args.number_of_samples)

    with open(metadata_path, 'rb') as metadata_fd:
        print 'loading metadata...'
        metadatas = pickle.load(metadata_fd)

    # with open(assets.get_normalization_cache_path(args.model), 'rb') as normalization_fd:
    #     print 'load normalizer from:', assets.get_normalization_cache_path(args.model)
    #     video_normalizer = pickle.load(normalization_fd)
    #
    # video_normalizer.normalize(vid)

    dp = DataProcessor(25, 16000)
    network = SpeechEnhancementNetwork.load(assets.get_model_cache_path(args.model))

    print 'predicting enhanced specs'
    print 'vid shape', vid.shape
    print 'spec shape', mix_specs.shape
    enhanced_specs = network.predict(np.swapaxes(mix_specs, 1, 2), vid)
    enhanced_specs = np.swapaxes(enhanced_specs, 1, 2)

    np.save('/cs/grad/asaph/testing/specs3.npy', enhanced_specs)

    mse  = np.mean(np.sum(np.square(source_specs.flatten() - enhanced_specs.flatten())))
    mae  = np.mean(np.sum(np.abs(source_specs.flatten() - enhanced_specs.flatten())))
    rmse = np.sqrt(np.mean(np.sum(np.square(source_specs.flatten() - enhanced_specs.flatten()))))
    mean_mean  = np.mean(np.square(source_specs.flatten() - enhanced_specs.flatten()))

    print 'time bins:', source_specs.shape[2]
    print 'mse loss:', mse
    print 'mae loss:', mae
    print 'rmse loss:', rmse
    print 'mean mean loss:', mean_mean

    date_dir = os.path.join(assets.get_prediction_dir(args.model), datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    os.mkdir(date_dir)

    for i in range(enhanced_specs.shape[0]):
        metadata = metadatas[i]
        audio_name = os.path.basename(os.path.splitext(metadata.audio_path)[0])
        noise_name = os.path.basename(os.path.splitext(metadata.noise_path)[0])

        print 'saving', audio_name, noise_name

        sample_dir = os.path.join(date_dir, audio_name + '_' + noise_name)
        os.mkdir(sample_dir)

        enhanced = dp.reconstruct_signal(enhanced_specs[i], mixed_phases[i])
        mixed = dp.reconstruct_signal(mix_specs[i], mixed_phases[i])
        source = AudioSignal(source_waveforms[i].astype(np.int16), metadata.audio_sampling_rate)

        source_audio_path = os.path.join(sample_dir, 'source.wav')
        mixed_audio_path = os.path.join(sample_dir, 'mixed.wav')
        enhanced_audio_path = os.path.join(sample_dir, 'enhanced.wav')

        source.save_to_wav_file(source_audio_path)
        mixed.save_to_wav_file(mixed_audio_path)
        enhanced.save_to_wav_file(enhanced_audio_path)

        video_extension = os.path.splitext(os.path.basename(metadata.video_path))[1]
        mixture_video_path = os.path.join(sample_dir, 'mixed' + video_extension)
        enhanced_speech_video_path = os.path.join(sample_dir, 'enhanced' + video_extension)

        ffmpeg.merge(metadata.video_path, mixed_audio_path, mixture_video_path)
        ffmpeg.merge(metadata.video_path, enhanced_audio_path, enhanced_speech_video_path)

class PredictionStorage(object):

    def __init__(self, storage_dir):
        self.__base_dir = os.path.join(storage_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
        os.mkdir(self.__base_dir)

    def __create_speaker_dir(self, speaker_id):
        speaker_dir = os.path.join(self.__base_dir, speaker_id)

        if not os.path.exists(speaker_dir):
            os.mkdir(speaker_dir)

        return speaker_dir

    def save_prediction(self, speaker_id, video_file_path, noise_file_path, speech_file_path,
                        mixed_signal, predicted_speech_signal, speech_spec):

        speaker_dir = self.__create_speaker_dir(speaker_id)

        speech_name = os.path.splitext(os.path.basename(video_file_path))[0]
        noise_name = os.path.splitext(os.path.basename(noise_file_path))[0]

        sample_prediction_dir = os.path.join(speaker_dir, speech_name + '_' + noise_name)
        os.mkdir(sample_prediction_dir)

        mixture_audio_path = os.path.join(sample_prediction_dir, 'mixture.wav')
        enhanced_speech_audio_path = os.path.join(sample_prediction_dir, 'enhanced.wav')
        source_speech_new_audio_path = os.path.join(sample_prediction_dir, 'source.wav')
        copy2(speech_file_path, source_speech_new_audio_path)

        mixed_signal.save_to_wav_file(mixture_audio_path)
        predicted_speech_signal.save_to_wav_file(enhanced_speech_audio_path)

        video_extension = os.path.splitext(os.path.basename(video_file_path))[1]
        mixture_video_path = os.path.join(sample_prediction_dir, 'mixture' + video_extension)
        enhanced_speech_video_path = os.path.join(sample_prediction_dir, 'enhanced' + video_extension)

        ffmpeg.merge(video_file_path, mixture_audio_path, mixture_video_path)
        ffmpeg.merge(video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)

        # os.unlink(mixture_audio_path)
        # os.unlink(enhanced_speech_audio_path)

        return sample_prediction_dir

def list_speakers(speakers, ignored_speakers, dataset_dir):
    if speakers is None:
        dataset = AudioVisualDataset(dataset_dir)
        speaker_ids = dataset.list_speakers()
    else:
        speaker_ids = speakers

    if ignored_speakers is not None:
        for speaker_id in ignored_speakers:
            speaker_ids.remove(speaker_id)

    return speaker_ids


def list_data(dataset_dir, speaker_ids, noise_dirs, max_files=None, shuffle=True):
    # if max_files is None:
    # 	max_files = 1000
    speech_dataset = AudioVisualDataset(dataset_dir)
    speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle)

    noise_dataset = AudioDataset(noise_dirs)
    noise_file_paths = noise_dataset.subset(shuffle=shuffle)

    # n_files = min(len(speech_subset), len(noise_file_paths))

    return speech_subset, noise_file_paths


def load_preprocessed_samples(preprocessed_blob_paths, max_samples=None):
    if type(preprocessed_blob_paths) is not list:
        preprocessed_blob_paths = [preprocessed_blob_paths]

    all_video_samples = []
    all_mixed_spectrograms = []
    all_source_spectrograms = []
    all_source_phases = []
    all_mixed_phases = []
    all_waveforms = []

    for preprocessed_blob_path in preprocessed_blob_paths:
        print('loading preprocessed samples from %s' % preprocessed_blob_path)

        with np.load(preprocessed_blob_path) as data:
            all_video_samples.append(data['video_samples'][:max_samples])
            all_mixed_spectrograms.append(data['mixed_spectrograms'][:max_samples])
            all_source_spectrograms.append(data['source_spectrograms'][:max_samples])
            all_mixed_phases.append(data['mixed_phases'][:max_samples])
            all_source_phases.append(data['source_phases'][:max_samples])
            all_waveforms.append(data['source_waveforms'][:max_samples])

    video_samples = np.concatenate(all_video_samples, axis=0)
    mixed_spectrograms = np.concatenate(all_mixed_spectrograms, axis=0)
    source_spectrograms = np.concatenate(all_source_spectrograms, axis=0)
    source_phases = np.concatenate(all_source_phases, axis=0)
    mixed_phases = np.concatenate(all_mixed_phases, axis=0)
    source_waveforms = np.concatenate(all_waveforms, axis=0)

    return (
        video_samples,
        mixed_spectrograms,
        source_spectrograms,
        source_phases,
        mixed_phases,
        source_waveforms
    )


class AssetManager:

    def __init__(self, base_dir):
        self.__base_dir = base_dir

        self.__cache_dir = os.path.join(self.__base_dir, 'cache')
        if not os.path.exists(self.__cache_dir):
            os.mkdir(self.__cache_dir)

        self.__preprocessed_dir = os.path.join(self.__cache_dir, 'preprocessed')
        if not os.path.exists(self.__preprocessed_dir):
            os.mkdir(self.__preprocessed_dir)

        self.__models_dir = os.path.join(self.__cache_dir, 'models')
        if not os.path.exists(self.__models_dir):
            os.mkdir(self.__models_dir)

        self.__out_dir = os.path.join(self.__base_dir, 'out')
        if not os.path.exists(self.__out_dir):
            os.mkdir(self.__out_dir)

        self.__data_dir = os.path.join(self.__base_dir, 'data')
        if not os.path.exists(self.__data_dir):
            os.mkdir(self.__data_dir)

    def get_data_set_dir(self, data_set, group='train'):
        return os.path.join(self.__data_dir, data_set, group)

    # @staticmethod
    # def get_noise_dirs(noise_dirs, group='train'):
    #     return [os.path.join(noise_dir, group) for noise_dir in noise_dirs]

    def create_preprocessed_data_dir(self, data_name):
        preprocessed_data_dir = os.path.join(self.__preprocessed_dir, data_name)
        if not os.path.exists(preprocessed_data_dir):
            os.mkdir(preprocessed_data_dir)

    def get_preprocessed_blob_data_path(self, data_name):
        return os.path.join(self.__preprocessed_dir, data_name, 'data.npz')

    def get_preprocessed_blob_metadata_path(self, data_name):
        return os.path.join(self.__preprocessed_dir, data_name, 'metadata.pkl')

    def create_model(self, model_name):
        model_dir = os.path.join(self.__models_dir, model_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

    def get_model_cache_path(self, model_name):
        return os.path.join(self.__models_dir, model_name)

    def get_normalization_cache_path(self, model_name):
        model_dir = os.path.join(self.__models_dir, model_name)
        return os.path.join(model_dir, 'normalization.pkl')

    def get_tensorboard_dir(self, model_name):
        model_dir = os.path.join(self.__models_dir, model_name)
        tensorboard_dir = os.path.join(model_dir, 'tensorboard')

        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)

        return tensorboard_dir

    def create_prediction_dir(self, model_name):
        pred_dir = os.path.join(self.__out_dir, model_name)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)

    def get_prediction_dir(self, model_name):
        return os.path.join(self.__out_dir, model_name)

    def create_prediction_storage(self, model_name, data_name):
        prediction_dir = os.path.join(self.__out_dir, model_name, data_name)
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        return prediction_dir


def main():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-bf', '--base_folder', type=str, default=BASE_FOLDER)
    parser.add_argument('-d', '--debug', type=int, default=0)

    action_parsers = parser.add_subparsers()

    preprocess_parser = action_parsers.add_parser('preprocess')
    preprocess_parser.add_argument('-dn', '--data_name', type=str, required=True)
    preprocess_parser.add_argument('-ds', '--dataset', type=str, required=True)
    preprocess_parser.add_argument('-s', '--speakers', nargs='+', type=str)
    preprocess_parser.add_argument('-is', '--ignored_speakers', nargs='+', type=str)
    preprocess_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
    preprocess_parser.add_argument('-ns', '--number_of_samples', type=int)
    preprocess_parser.add_argument('-c', '--cpus', type=int, default=8)
    preprocess_parser.set_defaults(func=preprocess, which='preprocess')

    train_gen_parser = action_parsers.add_parser('train_gen')
    train_gen_parser.add_argument('-mn', '--model', type=str, required=True)
    train_gen_parser.add_argument('-ds', '--dataset_dir', type=str, required=True)
    train_gen_parser.add_argument('-vds', '--val_dataset_dir', type=str)
    train_gen_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
    train_gen_parser.add_argument('-vn', '--val_noise_dirs', nargs='+', type=str)
    train_gen_parser.add_argument('-ts', '--train_speakers', nargs='+', type=str)
    train_gen_parser.add_argument('-vs', '--val_speakers', nargs='+', type=str)
    train_gen_parser.add_argument('-tis', '--train_ignored_speakers', nargs='+', type=str)
    train_gen_parser.add_argument('-vis', '--val_ignored_speakers', nargs='+', type=str)
    train_gen_parser.add_argument('-ns', '--number_of_samples', type=int)
    train_gen_parser.add_argument('-g', '--gpus', type=int, default=1)
    train_gen_parser.set_defaults(func=train)

    # train_parser = action_parsers.add_parser('train')
    # train_parser.add_argument('-mn', '--model', type=str, required=True)
    # train_parser.add_argument('-tdn', '--train_data_names', nargs='+', type=str, required=True)
    # train_parser.add_argument('-vdn', '--val_data_names', nargs='+', type=str, required=True)
    # train_parser.add_argument('-ns', '--number_of_samples', type=int)
    # train_parser.add_argument('-g', '--gpus', type=int, default=1)
    # train_parser.set_defaults(func=train)

    predict_parser = action_parsers.add_parser('predict')
    predict_parser.add_argument('-mn', '--model', type=str, required=True)
    predict_parser.add_argument('-dn', '--data_name', type=str, required=True)
    predict_parser.add_argument('-ns', '--number_of_samples', type=int)
    predict_parser.add_argument('-g', '--gpus', type=int, default=1)
    predict_parser.set_defaults(func=predict)


    args = parser.parse_args()

    if args.debug:
        sys.path.append('/usr/local/jetbrains/pycharm-2017.2.2/debug-eggs/pycharm-debug.egg')
        import pydevd

        pydevd.settrace('cooper-01', port=12345, stdoutToServer=True, stderrToServer=True)

    args.func(args)


if __name__ == '__main__':
    main()
