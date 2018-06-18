
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import *
from mediaio.audio_io import AudioSignal
import librosa as lb
from utils import *
from dataset import AudioVisualEntry

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
            # all_video_samples.append(data['video_samples'][:max_samples])
            all_mixed_spectrograms.append(data['mixed_spectrograms'][:max_samples])
            all_source_spectrograms.append(data['source_spectrograms'][:max_samples])
            all_mixed_phases.append(data['mixed_phases'][:max_samples])
            all_source_phases.append(data['source_phases'][:max_samples])
            all_waveforms.append(data['source_waveforms'][:max_samples])

    # video_samples = np.concatenate(all_video_samples, axis=0)
    mixed_spectrograms = np.concatenate(all_mixed_spectrograms, axis=0)
    source_spectrograms = np.concatenate(all_source_spectrograms, axis=0)
    source_phases = np.concatenate(all_source_phases, axis=0)
    mixed_phases = np.concatenate(all_mixed_phases, axis=0)
    source_waveforms = np.concatenate(all_waveforms, axis=0)

    return (
        # video_samples,
        mixed_spectrograms,
        source_spectrograms,
        source_phases,
        mixed_phases,
        source_waveforms
    )

sp_path = '/cs/grad/asaph/clean_sounds/pbai6a_s2.wav'
vid_path = '/cs/grad/asaph/clean_sounds/pbai6a_s2.mpg'
n_path = '/cs/grad/asaph/clean_sounds/geese.wav'

dp = DataProcessor(25, 16000)

entry = AudioVisualEntry('s2', sp_path, vid_path)

tup1 = dp.generate_batch_from_sample(entry, n_path)
tup2 = dp.preprocess_sample(entry, n_path)



s = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/pbai6a_s2.wav')
# g = AudioSignal.from_wav_file('/cs/grad/asaph/clean_sounds/geese.wav')
#
# mix_specs, source_specs, source_phases, mixed_phases, source_waveforms = load_preprocessed_samples(
# 	'/cs/labs/peleg/asaph/playground/avse/cache/preprocessed/obama-libri-test/data.npz', max_samples=2)
#
#
#
#
#
# spec, phase = dp.get_mag_phase(s.get_data())
# new = dp.reconstruct_signal(np.round(mix_specs[0]), mixed_phases[0])
# new2 = dp.reconstruct_signal(spec, phase)
#
# new.save_to_wav_file('/cs/grad/asaph/testing/obama2.wav')
# new2.save_to_wav_file('/cs/grad/asaph/testing/grid2.wav')

# while g.get_number_of_samples() < s.get_number_of_samples():
# 	g = AudioSignal.concat([g, g])
#
# g.truncate(s.get_number_of_samples())
#
#
# s = s.get_data().astype('f')
# g = g.get_data().astype('f')
#
#
# mag_s, phase_s = lb.magphase(lb.stft(s, 640, 160))
# mag_g, phase_g = lb.magphase(lb.stft(g, 640, 160))
#
# s_plus_g = lb.istft(mag_s * phase_g).astype(np.int16)
# g_plus_s = lb.istft(mag_g * phase_s).astype(np.int16)
# s = lb.istft(mag_s * phase_s).astype(np.int16)
# g = lb.istft(mag_g * phase_g).astype(np.int16)
#
# AudioSignal(s_plus_g, 16000).save_to_wav_file('/cs/grad/asaph/testing/s_g.wav')
# AudioSignal(g_plus_s, 16000).save_to_wav_file('/cs/grad/asaph/testing/g_s.wav')
# AudioSignal(s, 16000).save_to_wav_file('/cs/grad/asaph/testing/s_s.wav')
# AudioSignal(g, 16000).save_to_wav_file('/cs/grad/asaph/testing/g_g.wav')




# raw_data = s.get_data().astype('f')
#
# pl.plot(raw_data)
#
# data = np.abs(raw_data)
# # pl.figure()
# # pl.plot(data)
#
# k = 25.
#
# med = medfilt(data, kernel_size=int(k))
# mean = np.convolve(data, np.arange(k) / k, mode='same')
#
#
# pl.figure()
# pl.plot(mean)
# # pl.figure()
# # pl.plot(med)
#
# thresh = 10000
#
# line = (mean > thresh) * 10000
# pl.figure(1)
# pl.plot(line, 'r')
#
# pl.show()



