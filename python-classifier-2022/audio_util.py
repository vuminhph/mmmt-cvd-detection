import os
import numpy as np
import random
import scipy as sp
from scipy.signal import butter, lfilter
from tqdm.notebook import tqdm
import torch
import torchaudio
from torchaudio import transforms

from Const import *

class AudioUtil():
    # Bandpass filter
    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = AudioUtil.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def schmidt_spike_removal(original_signal, fs):
        """
        This function removes the spikes in a signal as done by Schmidt et al in
        the paper:
        Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
        (2010). Segmentation of heart sound recordings by a duration-dependent
        hidden Markov model. Physiological Measurement, 31(4), 513-29.
        
        The spike removal process works as follows:
        (1) The recording is divided into 500 ms windows.
        (2) The maximum absolute amplitude (MAA) in each window is found.
        (3) If at least one MAA exceeds three times the median value of the MAA's,
        the following steps were carried out. If not continue to point 4.
        (a) The window with the highest MAA was chosen.
        (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
        (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
        (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
        (e) The defined noise spike was replaced by zeroes.
        (f) Resume at step 2.
        (4) Procedure completed.
        
        Inputs:
        - original_signal: The original (1D) audio signal array
        - fs: the sampling frequency (Hz)
        
        Outputs:
        - despiked_signal: the audio signal with any spikes removed.
        """

        no_sweeps = 0

        # Find the window size (500 ms)
        window_size = round(fs/2)

        cur_signal = np.copy(original_signal)
        # Find any samples outside of an integer number of windows
        trailing_samples = len(cur_signal) % window_size

        # Reshape the original signal into a number of windows
        sample_frames = np.reshape(cur_signal[:len(cur_signal)-trailing_samples], [-1, window_size])

        # Find the MAAs
        MAAs = np.amax(abs(sample_frames), axis=1)

        # While there are still samples greater than 3 * the median value of MAAs, remove those spikes
        while any(maa > 3 * np.median(MAAs) for maa in MAAs) and no_sweeps <= MAX_SWEEPS:
            no_sweeps += 1
            # Find the window with the max MAA
            val = np.amax(MAAs)
            window_num = np.where(MAAs == val)[0][0]

            # Find the position of the spike within that window
            spiked_frame = sample_frames[:, window_num] 

            val = np.amax(abs(spiked_frame))
            spike_position = np.where(abs(spiked_frame) == val)[0][0]

            # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative)
            zero_crossings = np.where(np.diff(np.sign(spiked_frame)))[0] # index

            # Find the start of the spike, finding the last zero crossing before spike position. If that is empty, take the start of the window:
            try:
                spike_start = zero_crossings[np.nanargmax(np.where(zero_crossings < spike_position,zero_crossings,np.nan))]
            except:
                spike_start = 0
            # Find the end of the spike, finding the first zero crossing after spike position, If that is empty, take the end of the window
            try:
                spike_end = zero_crossings[np.nanargmin(np.where(zero_crossings > spike_position,zero_crossings,np.nan))]
            except:
                spike_end = len(spiked_frame) - 1

            # Set to zero
            sample_frames[spike_start : spike_end + 1, window_num] = 0.0001   

            # Recalculate MAAs  
            MAAs = np.amax(abs(sample_frames), axis=1)

        despiked_signal = sample_frames.flatten()
        # Add the trailing samples back to the signal
        despiked_signal = np.hstack((despiked_signal, cur_signal[len(despiked_signal) :]))

        return despiked_signal

    @staticmethod
    def audio_norm(signal):
        norm_signal = signal / np.max(np.abs(signal),axis=0)
        return norm_signal

    @staticmethod
    def load_wav_file(filename):
        """
        Load and preprocess a wav file
        """
        frequency, signal = sp.io.wavfile.read(filename)
        new_samps = round(len(signal) / frequency * NEW_SAMPLING_RATE)
        signal = sp.signal.resample(signal, new_samps)

        # Band pass filter between 25-400 Hz
        signal = AudioUtil.butter_bandpass_filter(signal, LOW_CUT_FREQ, HIGH_CUT_FREQ, NEW_SAMPLING_RATE)
        # Remove spikes in the recording
        signal = AudioUtil.schmidt_spike_removal(signal, NEW_SAMPLING_RATE)

        return signal, NEW_SAMPLING_RATE

    @staticmethod
    def load_wav_file_no_preprocessing(filename):
        frequency, recording = sp.io.wavfile.read(filename)
        new_samps = int(len(recording) / frequency * NEW_SAMPLING_RATE)
        recording = sp.signal.resample(recording, new_samps)

        return recording, NEW_SAMPLING_RATE

    @staticmethod
    def get_segmentation_file(recording_file_path):
        """
        Look for segmentation tsv file of the recording in the data_folder and return if found

        Argument: 
        - recording_file_name: full path of the recording file
        Returns:
        - The full path to the segmentation file

        """
        segmentation_file_path = recording_file_path.replace('wav', 'tsv')
        if os.path.exists(segmentation_file_path):
            return segmentation_file_path
        else:
            # TODO perform segmentation on the recording file if none exists
            return None

    @staticmethod
    def get_cardiac_states(segmentation_file_path):
        """
        Get the cardiac cycles info

        Arguments:
        - segmentation_file_path: path to the segmentation file
        Returns: 
        - cardiac_states: tuple of size 2 containing
                            + state_starts: an array of size 4xN: containing start times of s1, sys, s2, dia
                            + state_ends: an array of size 4xN: containing end times of s1, sys, s2, dia
        """
        state_starts = np.zeros((4, 1))
        state_ends = np.zeros((4, 1))
        cycle_idx = 0
        last_state = -1

        with open(segmentation_file_path, 'r') as f:
            seg_data = f.read()
            for l in seg_data.split('\n'):
                line_data = l.split('\t')
                if len(line_data) != 3:
                    continue

                state_s, state_e, state = [round(float(line_data[i]),3) if i < 2 else int(line_data[i]) for i in range(len(line_data))]
                
                if state not in range(1,5):
                    continue
                
                if state <= last_state:
                    cycle_idx += 1
                    state_starts = np.hstack((state_starts, np.zeros((4,1))))
                    state_ends = np.hstack((state_ends, np.zeros((4,1))))

                state_starts[state - 1, cycle_idx] = state_s
                state_ends[state - 1, cycle_idx] = state_e
                last_state = state

            if cycle_idx == 0 and (state_starts[:, 0] == np.zeros((4,1))).all():
                return None
                
            state_starts[state_starts == 0] = np.nan
            state_ends[state_ends == 0] = np.nan

        cardiac_states = (state_starts, state_ends)

        return cardiac_states

    @staticmethod
    def split_cardiac_cycles(recording, cardiac_states):
        """
        Split recording into cardiac cycles base on segmentation info

        Arguments: 
        - recording: signal array represent the recording, containing many cardiac cycles
        - cardiac_states: tuple of size 2 containing:
                + state_starts: an array of size 4xN: containing start times of s1, sys, s2, dia
                + state_ends: an array of size 4xN: containing end times of s1, sys, s2, dia
        
        Returns:
        - cardiac_cycles: list of Nx1 arrays, each represents a cardiac cycle
        """

        state_starts = cardiac_states[0]
        state_ends = cardiac_states[1]
        no_cycles = state_starts.shape[1]

        cardiac_cycles = []

        for i in range(no_cycles):
            try:
                # If not complete cycle then continue
                if np.any(np.isnan(state_starts[:, i])) or np.any(np.isnan(state_ends[:, i])):
                    continue
                
                cycle_start = int(state_starts[0, i] * NEW_SAMPLING_RATE)
                # cycle_end is the upper bound
                cycle_end = int(state_ends[3, i] * NEW_SAMPLING_RATE) + 1
            except:
                # tqdm.write(str(state_starts[: , i : i + 3]))
                pass

            new_cycle = recording[cycle_start : cycle_end]
            cardiac_cycles.append(new_cycle)

        return cardiac_cycles

    @staticmethod
    def pad_signal(signal, max_ms):
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
        """
        sig_len = signal.shape[0]
        max_len = round(max_ms / 1000 * NEW_SAMPLING_RATE)

        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = np.zeros((pad_begin_len))
        pad_end = np.zeros((pad_end_len))

        signal = np.concatenate((pad_begin, signal, pad_end))
        
        return signal

    @staticmethod
    def spectro_gram(signal, n_mels=32, n_fft=128, hop_len=None):
        """
        Generate a Spectrogram
        """

        top_db = 80

        signal = torch.tensor(signal, dtype=torch.float32)

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sample_rate=NEW_SAMPLING_RATE, f_max=HIGH_CUT_FREQ, f_min=LOW_CUT_FREQ, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec