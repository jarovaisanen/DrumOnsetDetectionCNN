# -*- coding: utf-8 -*-

import os
import os.path
import numpy as np
import pandas as pd
import random
from pathlib import Path

import librosa
import librosa.display
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import stft, spectrogram

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class DatasetGenerator:
    def __init__(self, 
                 label_set, 
                 sample_rate=44100,
                 channels=[2048],
                 mel_bands=80,
                 time_frames=15,
                 diff_from_onset_ms=0.030,
                 threshold_freq=15500,
                 drum_instrument='bd'):
        
        self.label_set = label_set
        self.sample_rate = sample_rate
        self.channels = channels
        self.mel_bands = mel_bands
        self.time_frames = time_frames
        self.diff_from_onset_ms = diff_from_onset_ms
        self.threshold_freq = threshold_freq
        self.drum_instrument = drum_instrument

    def apply_train_test_split(self, test_size, random_state):
        self.df_train, self.df_test = train_test_split(self.df, 
                                                       test_size=test_size,
                                                       random_state=random_state)

    def apply_train_val_split(self, val_size, random_state):
        self.df_train, self.df_val = train_test_split(self.df_train, 
                                                      test_size=val_size, 
                                                      random_state=random_state)

    def apply_train_test_split_by_windows(self, test_size, shuffle_train_data=True):
        self.df_train = self.df
        data, labels = self.get_data('train', shuffle_train_data=shuffle_train_data)
        
        larger_portion = int(len(data)*(1-test_size))
        train_data = data[:larger_portion]
        train_labels = labels[:larger_portion]
        test_data = data[larger_portion:]
        test_labels = labels[larger_portion:]

        # Remove effects of to_categorical function, convert to binary.
        test_labels = np.argmax(test_labels, axis=1)
        return train_data, train_labels, test_data, test_labels

    def load_datafiles(self, dir):
        files = list(Path(dir).rglob('*wav'))
        data = []

        # Loop over files to get samples.
        for file in files:
            wav_file = os.path.join(dir, file.name)
            filename_base = Path(wav_file).stem
            # Files with '_acc' contain only the accompaniment track.
            if '_acc' in filename_base:
                continue
            else:
                # NOTE: Modify the annotation file format and paths to suit your
                # needs.
                annotations_path = os.path.join(dir, filename_base + '.txt.' + self.drum_instrument)
            # If wav file has matching drum instrument annotation file, add to input data.
            if os.path.isfile(annotations_path):
                sample = (wav_file, annotations_path)
                data.append(sample)

        # Data Frames with wavs and matching annotation paths.
        df = pd.DataFrame(data, columns=['wav_file', 'annotations'])
        self.df = df
        return df

    def read_wav_file(self, x):
        # Read wavfile using scipy wavfile.read.
        _, wav = wavfile.read(x) 
        
        # Normalize.
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        
        wav_dim = np.shape(wav)
        if len(wav_dim) == 2:
            # Convert stereo to mono.
            wav = wav.sum(axis=1) / 2
        return wav

    def process_wav_file(self, wav_file, annotations_file, win_length=2048, eps=1e-10):
        # Read wav file to array
        wav = self.read_wav_file(wav_file)
        sample_rate = self.sample_rate
        
        hop_length = 441 # win_length // 4 # 2048 // 4 = 512
        noverlap = win_length - hop_length

        # Create stft using scipy.signal, which returns frequencies, timestamps,
        # and the STFT spectrogram, which can be passed to the melspectrogram.
        freqs, times, spec = spectrogram(wav, sample_rate, window='hann', nperseg=win_length, noverlap=noverlap, mode='complex')

        # Do harmonic-percussive decomposition. Result should be a power spectrogram.
        # Get a more isolated percussive component by widening its margin.
        _, S_percussive = librosa.decompose.hpss(spec, margin=(1.0, 5.0))

        S = librosa.feature.melspectrogram(S=np.abs(S_percussive), sr=sample_rate, window='hann', win_length=win_length, hop_length=hop_length, n_mels=self.mel_bands, center=False, fmax=self.threshold_freq)

        # Convert spectrogram from power scale to db. Makes the spectrogram clearer.
        S_db = librosa.core.power_to_db(S, ref=np.max)

        # Scale logarithmically. 
        # Left this out, seems to produce better results without it.
        # amp = np.log(np.abs(S_db)+eps)

        # Expands dimensions to comply the channel dimensionality. Currently
        # using a single channel, but could be 3 (channels based on STFT window
        # sizes 1024, 2048 and 4096, like channels in RGB images).
        S_expanded = np.expand_dims(S_db, axis=2)

        # Cut percussive spectrogram and index onset labels.
        spectrograms = []
        sp = SpectrogramProcessor(S_expanded, times, annotations_file)
        spectrograms = sp.split_spectrogram(self.time_frames)
        annotations = sp.read_annotations()
        onsets = sp.get_onsets(spectrograms, annotations, self.diff_from_onset_ms)
        spectrograms = [s[0] for s in spectrograms]  # Remove time indices.

        return spectrograms, onsets

    def get_data(self, mode, shuffle_train_data=True):
        if mode == 'train':
            df = self.df_train
            # Shuffle input data.
            audiofile_ids = random.sample(range(df.shape[0]), df.shape[0]) if shuffle_train_data else list(range(df.shape[0])) 
        elif mode == 'val':
            df = self.df_val
            audiofile_ids = list(range(df.shape[0]))
        elif mode == 'test':
            df = self.df_test
            audiofile_ids = list(range(df.shape[0]))
        else:
            raise ValueError('The mode should be either train, val or test.')        
        return self.get_singlechannel_data(df, audiofile_ids, mode == 'test')

    def get_singlechannel_data(self, df, audiofile_ids, is_test):
        input_data = []
        labels = []

        for i in range(0, len(audiofile_ids)):
            for win_length in self.channels:
                spectrograms, onsets = self.process_wav_file(df.wav_file.values[i], df.annotations.values[i], win_length=win_length)
                input_data.extend(spectrograms)
                labels.extend(onsets)
        
        # Convert to numpy array.
        input_data = np.array(input_data)

        if not is_test:
            # Process labels to one-hot encoding.
            labels = to_categorical(labels, num_classes=len(self.label_set))

        return input_data, labels

    
    def create_custom_mix_data(self, dirs):
        for dir in dirs:
            drum_files = list(Path(dir + '\\audio\\wet_mix').rglob('*minus-one*.wav'))

            # Loop over files to get samples.
            for drum_file in drum_files:
                filename = drum_file.name

                # Get corresponding accompaniment.
                acc_dir = dir + '\\audio\\accompaniment'
                acc_file = os.path.join(acc_dir, filename)
                
                wav_file = self.mix_tracks(drum_file, acc_file)
                result_file = dir + '\\audio\\mixed\\' + filename
                wav_file.export(result_file, format='wav')

    def mix_tracks(self, drums_wav, acc_wav):
        drums = AudioSegment.from_wav(drums_wav)
        acc = AudioSegment.from_wav(acc_wav)

        # Increase drums volume and decrease accompaniment volume.
        drums_vol = drums + 6
        acc_vol = acc - 6

        combined = drums_vol.overlay(acc_vol)
        return combined


class SpectrogramProcessor:
    """Processes input spectrogram for to be used with a neural network.

    Construct with:
        - spectrogram created from an input audio file
        - time_indices (timestamps) of the input spectrogram
        - ground truth onset annotations (timestamps)
    """
    def __init__(self, spectrogram, time_indices, annotations_file):
        self.spectrogram = spectrogram
        self.time_indices = time_indices
        self.annotations_file = annotations_file

    def read_annotations(self):
        """Read annotations from text file line by line. 
           If required, split the drum instrument tag from the timestamp
           Would be like: 2.35 bd
                          3.47 sd
        """
        with open(self.annotations_file) as file:
            annotations = [float(line.rstrip()) for line in file]
        return annotations

    def split_spectrogram(self, split_size):
        """Splits an input spectrogram to subsequent smaller spectrograms based
        on the time dimension size.

        Parameters
        ----------
        split_size : int 
            Size of the subsequent time windows.

        Returns
        -------
        spectrograms : numpy array 
            Split spectrograms with the centered time value.

        """
        spectrogram = self.spectrogram
        time_indices = self.time_indices
        spectrograms = []

        for i in range(len(time_indices)):
            time = time_indices[i]
            # Take every mel band and e.g. 15 time frames, and 15 next at the
            # next iteration.
            s = spectrogram[:, i:i+split_size]
            if s.shape[1] == split_size:
                spectrograms.append((s, time))
        return spectrograms

    def get_onsets(self, spectrograms, annotations, diff_from_onset_ms):
        """Maps annotated onsets to given spectrograms based on the time indices
        of the audio signal.

        Parameters
        ----------
        spectrograms : numpy array 
            Array consisting of tuples, that contain split spectrograms (each
            the same size) and a time frame value associated with each spectrogram.
        annotations : numpy array 
            Actual ground truth onset annotation time labels.'
        diff_from_onset_ms : float
            Acceptable onset difference from the ground truth in milliseconds.
            Default is 30 ms. 30 ms was used in Automatic Drum Transcription with CNN article.

        Returns
        -------
        onsets : numpy array 
            Labels of onsets, same sized array as input spectrograms.

        """
        onsets = []

        # If there are no annotations for given audio file, mark every onset as
        # nonset.
        if not annotations:
            onsets = [0] * len(spectrograms)
            return onsets

        # Loop through spectrograms and label them with onset or no-onset based
        # on annotations. Use time stamps to determine spectrogram time frame
        # location. Assert that onsets.size equals spectrograms.size.

        current_index = 0
        for annotation in annotations:
            for i in range(current_index, len(spectrograms)):
                spectrogram, time = spectrograms[i]
                # A detected onset is considered correct if the absolute time difference
                # with the associated ground truth onset does not exceed 30 ms.
                #
                # O = onset (ground truth)
                # x = time value
                # |----O----| = acceptable onset zone within the ground truth
                #
                # --x----------|--x--O----x|---------x--->
                #                                       t-axis
                #
                # Take more of the negative window, because variable 'time' is
                # on the beginning of the spectrogram. Basically align the onset
                # zone closer to the middle.
                if ((annotation - diff_from_onset_ms) <= time) and (time <= annotation):
                    onsets.append(1)
                    
                    # If we are at the last annotation, don't come out of the
                    # inner loop before finishing.
                    if annotation == annotations[-1]:
                        continue
                # Break to the next annotation only if we are sure there aren't
                # any more onsets to be marked for the current ground truth
                # annotation. That is, we have gone past the current annotation
                # zone (and we are not at the last annotation).
                elif ((annotation + diff_from_onset_ms) < time) and (annotation != annotations[-1]):
                    onsets.append(0)
                    break
                else:
                    onsets.append(0)
            current_index = i + 1

        # Make sure onsets are same size as spectrograms.
        assert len(spectrograms) == len(onsets)
        return onsets
