from os.path import join
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import torch
from python_speech_features import mfcc
from tqdm.notebook import tqdm

from helper_code import *
from Const import *
from audio_util import AudioUtil

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)
    
# Extract clinical features from the data.
def get_clinical_features(data):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))
    features = torch.Tensor(features)

    return features

# Extract features from the data.
def get_full_features(data, recordings, recording_file_paths):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    features_len = 124

    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, features_len + 1), dtype=np.float32)
    num_locations = len(locations)
    num_recordings = len(recordings)

    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    pcg_features = get_PCG_features(recordings[i], recording_file_paths[i])
                    if pcg_features is not None:
                        recording_features[j, 0] = 1
                        recording_features[j, 1:] = pcg_features

                    # recording_features[j, 1] = np.nanmean(recordings[i])
                    # recording_features[j, 2] = np.var(recordings[i])
                    # recording_features[j, 3] = skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))
    features[abs(features) == np.inf] = np.nan

    return np.asarray(features, dtype=np.float32)


# New functions #
def get_PCG_features(recording, recording_file_path):
    """
    Extract the time domain and frequency domain features from a recording file

    Arguments:
     - recordings: the preprocessed recording signal
     - recording_file_path: the path to the recording file
    Returns: 
     - features: the extracted features array (36 + 88)
    """
    if len(recording) <= 0:
        return None
    
    # tqdm.write(f"Extracting features from {recording_file_path}...")

    # Extract time domain features
    segmentation_file_path = AudioUtil.get_segmentation_file(recording_file_path)

    cardiac_states = AudioUtil.get_cardiac_states(segmentation_file_path)

    # Extract time domain features
    time_domain_features = extract_time_domain_features(cardiac_states, recording)
    # Extract frequency domain features
    frequency_domain_features = extract_frequency_domain_features(cardiac_states, recording)

    features = np.hstack((time_domain_features, frequency_domain_features))

    return np.asarray(features, dtype=np.float32)

def extract_time_domain_features(state_data, signal):
    """
    Extract features related to recording's time domain
    
    Arguments:
    - State_data: a tuple containing 2 elements:
        + state_starts: containing every state start time
        + state_ends: containing every state end time
        Each a 4xN matrix, each row corresponding to a state, the columns represents heart cycles
    
    - signal: 1xN ndarray containing the recording signal

    Returns:
     - time_domain_features: An array containing the extracted features (16 + 20)
    """

    # Interval features
    state_starts = state_data[0]
    state_ends = state_data[1]
    no_cycles = state_starts.shape[1]
    # print(f"Number of heart cycles detected: {no_cycles}")

    absolute_signal = abs(signal)
    # print(f"Signal length: {len(signal)}")

    sys_s1_amp_ratios = []
    dia_s2_amp_ratios = []
    s1_skewness = []
    s2_skewness = []
    sys_skewness = []
    dia_skewness = []
    s1_kurtosis = []
    s2_kurtosis = []
    sys_kurtosis = []
    dia_kurtosis = []

    # RR intervals features
    RR_intervals = []
    for i in range(no_cycles):
        # RR intervals
        if i != no_cycles - 1:
            RR_intervals.append(state_starts[0,i + 1] - state_starts[0,i])

        mean_abs_sys = None
        mean_abs_s1 = None
        mean_abs_dia = None
        mean_abs_s2 = None
        
        if not np.isnan(state_starts[1,i]) and not np.isnan(state_ends[1,i]):
            sys_index_start = int(state_starts[1,i] * NEW_SAMPLING_RATE)
            sys_index_end = int(state_ends[1,i] * NEW_SAMPLING_RATE)
            
            mean_abs_sys = np.nanmean(absolute_signal[sys_index_start:sys_index_end+1])
            # Skewness
            sys_skewness.append(skew(signal[sys_index_start:sys_index_end+1], nan_policy='omit'))
            # Kurtosis
            sys_kurtosis.append(kurtosis(signal[sys_index_start:sys_index_end+1], nan_policy='omit'))
        
        if not np.isnan(state_starts[0,i]) and not np.isnan(state_ends[0,i]):
            s1_index_start = int(state_starts[0,i] * NEW_SAMPLING_RATE)
            s1_index_end = int(state_ends[0,i] * NEW_SAMPLING_RATE)

            mean_abs_s1 = np.nanmean(absolute_signal[s1_index_start:s1_index_end+1])
            # Skewness
            s1_skewness.append(skew(signal[s1_index_start:s1_index_end+1], nan_policy='omit'))
            # Kurtosis 
            s1_kurtosis.append(kurtosis(signal[s1_index_start:s1_index_end+1], nan_policy='omit'))

        # Ratio of mean abs value between sys and s1 in each cycle
        if mean_abs_s1 and mean_abs_sys:
            sys_s1_amp_ratios.append(mean_abs_sys / mean_abs_s1)

        if not np.isnan(state_starts[3,i]) and not np.isnan(state_ends[3,i]):
            dia_index_start = int(state_starts[3,i] * NEW_SAMPLING_RATE)
            dia_index_end = int(state_ends[3,i] * NEW_SAMPLING_RATE)

            mean_abs_dia = np.nanmean(absolute_signal[dia_index_start:dia_index_end+1])
            # Skewness
            dia_skewness.append(skew(signal[dia_index_start:dia_index_end+1], nan_policy='omit'))
            # Kurtosis
            dia_kurtosis.append(kurtosis(signal[dia_index_start:dia_index_end+1], nan_policy='omit'))
        
        if not np.isnan(state_starts[2,i]) and not np.isnan(state_ends[2,i]):
            s2_index_start = int(state_starts[2,i] * NEW_SAMPLING_RATE)
            s2_index_end = int(state_ends[2,i] * NEW_SAMPLING_RATE)
        
            mean_abs_s2 = np.nanmean(absolute_signal[s2_index_start:s2_index_end+1])
            # Skewness
            s2_skewness.append(skew(signal[s2_index_start:s2_index_end+1], nan_policy='omit'))
            # Kurtosis
            s2_kurtosis.append(kurtosis(signal[s2_index_start:s2_index_end+1], nan_policy='omit'))

        # Ratio of mean abs value between dia and s2 in each cycle
        if mean_abs_dia and mean_abs_s2:
            dia_s2_amp_ratios.append(mean_abs_dia / mean_abs_s2)


    RR_features = [np.nanmean(RR_intervals), np.nanstd(RR_intervals)]

    # State interval features 
    state_intervals = state_ends - state_starts
    # S1 intervals
    s1_features = [np.nanmean(state_intervals[0, :]), np.nanstd(state_intervals[0, :])]
    # systolic intervals
    sys_features = [np.nanmean(state_intervals[1, :]), np.nanstd(state_intervals[1, :])]
    # S2 intervals
    s2_features = [np.nanmean(state_intervals[2, :]), np.nanstd(state_intervals[2, :])]
    # diastolic intervals
    dia_features = [np.nanmean(state_intervals[3, :]), np.nanstd(state_intervals[3, :])]
    # Systole to RR interval ratios 
    sys_rr_ratios = state_intervals[1, :-1] / np.array((RR_intervals))
    sys_rr_ratios_features = [np.nanmean(sys_rr_ratios), np.nanstd(sys_rr_ratios)]
    # Diastole to RR interval ratios 
    dia_rr_ratios = state_intervals[3, :-1] / np.array((RR_intervals))
    dia_rr_ratios_features = [np.nanmean(dia_rr_ratios), np.nanstd(dia_rr_ratios)]
    # Systole to Diastole interval ratios
    sys_dia_ratios = state_intervals[1, :] / state_intervals[3, :]
    sys_dia_ratios_features = [np.nanmean(sys_dia_ratios), np.nanstd(sys_dia_ratios)]

    interval_features = np.hstack((RR_features, s1_features, sys_features, s2_features, dia_features, sys_rr_ratios_features, dia_rr_ratios_features, sys_dia_ratios_features))
    # print(interval_features)
    # print(interval_features.shape)

    # Amplitude features
    sys_s1_amp_ratios = np.array((sys_s1_amp_ratios))
    sys_s1_amp_ratios_features = [np.nanmean(sys_s1_amp_ratios), np.nanstd(sys_s1_amp_ratios)]
    
    dia_s2_amp_ratios = np.array((dia_s2_amp_ratios))
    dia_s2_amp_ratios_features = [np.nanmean(dia_s2_amp_ratios), np.nanstd(dia_s2_amp_ratios)]

    s1_skewness = np.array((s1_skewness))
    s1_skewness_features = [np.nanmean(s1_skewness), np.nanstd(s1_skewness)]
    
    s2_skewness = np.array((s2_skewness))
    s2_skewness_features = [np.nanmean(s2_skewness), np.nanstd(s2_skewness)]
    
    sys_skewness = np.array((sys_skewness))
    sys_skewness_features = [np.nanmean(sys_skewness), np.nanstd(sys_skewness)]
    
    dia_skewness = np.array((dia_skewness))
    dia_skewness_features = [np.nanmean(dia_skewness), np.nanstd(dia_skewness)]
    
    s1_kurtosis = np.array((s1_kurtosis))
    s1_kurtosis_features = [np.nanmean(s1_kurtosis), np.nanstd(s1_kurtosis)]
    
    s2_kurtosis = np.array((s2_kurtosis))
    s2_kurtosis_features = [np.nanmean(s2_kurtosis), np.nanstd(s2_kurtosis)]
    
    sys_kurtosis = np.array((sys_kurtosis))
    sys_kurtosis_features = [np.nanmean(sys_kurtosis), np.nanstd(sys_kurtosis)]
    
    dia_kurtosis = np.array((dia_kurtosis))
    dia_kurtosis_features = [np.nanmean(dia_kurtosis), np.nanstd(dia_kurtosis)]

    amplitude_features = np.hstack((sys_s1_amp_ratios_features, dia_s2_amp_ratios_features, s1_skewness_features, s2_skewness_features, sys_skewness_features, dia_skewness_features, s1_kurtosis_features, s2_kurtosis_features, sys_kurtosis_features, dia_kurtosis_features))
    # print(amplitude_features)
    # print(amplitude_features.shape)
    
    time_domain_features = np.hstack((interval_features, amplitude_features))
    # print(time_domain_features)
    # print(time_domain_features.shape)

    if np.any(np.isnan(time_domain_features)):
        tqdm.write(f"Time: {time_domain_features}")

    return time_domain_features
    
def calculate_nfft(samplerate, winlen):
    """
    Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.

    Arguments: 
      - samplerate: The sample rate of the signal we are working with, in Hz.
      - winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def extract_frequency_domain_features(state_data, signal):
    """
    Extract features related to recording's frequency domain

    Arguments:
    - State_data: a tuple containing 2 elements:
        + state_starts: containing every state start time
        + state_ends: containing every state end time
        Each a 4xN matrix, each row corresponding to a state, the columns represents heart cycles
     - signal: 1xN ndarray containing the recording signal

    Returns:
     - frequency_domain_features: An array containing the extracted features (36 + 52)
    """
    state_starts = state_data[0]
    state_ends = state_data[1]

    no_cycles = state_starts.shape[1]

    # Get power spectrum for each frequency band
    bands_power_spectrum = [None for i in range(9)]
    bands_power_spectrum[0] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 25, 45, NEW_SAMPLING_RATE))) ** 2 
    bands_power_spectrum[1] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 45, 65, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[2] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 65, 85, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[3] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 85, 105, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[4] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 105, 125, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[5] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 125, 150, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[6] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 150, 200, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[7] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 200, 300, NEW_SAMPLING_RATE))) ** 2
    bands_power_spectrum[8] = abs(fft(AudioUtil.butter_bandpass_filter(signal, 300, 400, NEW_SAMPLING_RATE))) ** 2

    s1_med_power_spectrums = [[] for i in range(9)]
    sys_med_power_spectrums = [[] for i in range(9)]
    s2_med_power_spectrums = [[] for i in range(9)]
    dia_med_power_spectrums = [[] for i in range(9)]

    # Get MFCC features
    s1_mfccs = np.empty((no_cycles, 13))
    sys_mfccs = np.empty((no_cycles, 13))
    s2_mfccs = np.empty((no_cycles, 13))
    dia_mfccs = np.empty((no_cycles, 13))

    # print(f"MFCC: {mfcc(signal, NEW_SAMPLING_RATE)}")

    for i in range(no_cycles):
        
        if not np.isnan(state_starts[0,i]) and not np.isnan(state_ends[0,i]):
            # Get index
            s1_index_start = int(state_starts[0,i] * NEW_SAMPLING_RATE)
            s1_index_end = int(state_ends[0,i] * NEW_SAMPLING_RATE)

            # Save median power spectrum
            for j in range(9):
                s1_med_power_spectrums[j].append(np.median(bands_power_spectrum[j][s1_index_start:s1_index_end]))
            
            # Save MFCC features
            win_len = state_ends[0,i] - state_starts[0,i] + 0.001
            nfft = calculate_nfft(NEW_SAMPLING_RATE, win_len)
            s1_mfccs[i, :] = mfcc(signal[s1_index_start:s1_index_end+1], NEW_SAMPLING_RATE, winlen=win_len, winfunc=np.hamming, nfft=nfft)[0]
        
        if not np.isnan(state_starts[1,i]) and not np.isnan(state_ends[1,i]):
            # Get index
            sys_index_start = int(state_starts[1,i] * NEW_SAMPLING_RATE)
            sys_index_end = int(state_ends[1,i] * NEW_SAMPLING_RATE)

            # Save median power spectrum
            for j in range(9):
                sys_med_power_spectrums[j].append(np.median(bands_power_spectrum[j][sys_index_start:sys_index_end]))

            # Save MFCC features
            win_len = state_ends[1,i] - state_starts[1,i] + 0.001
            nfft = calculate_nfft(NEW_SAMPLING_RATE, win_len)
            sys_mfccs[i, :] = mfcc(signal[sys_index_start:sys_index_end+1], NEW_SAMPLING_RATE, winlen=win_len, winfunc=np.hamming, nfft=nfft)[0]
        
        if not np.isnan(state_starts[2,i]) and not np.isnan(state_ends[2,i]):
            # Get index
            s2_index_start = int(state_starts[2,i] * NEW_SAMPLING_RATE)
            s2_index_end = int(state_ends[2,i] * NEW_SAMPLING_RATE)

            # Save median power spectrum
            for j in range(9):
                s2_med_power_spectrums[j].append(np.median(bands_power_spectrum[j][s2_index_start:s2_index_end]))

            # Save MFCC features
            win_len = state_ends[2,i] - state_starts[2,i] + 0.001
            nfft = calculate_nfft(NEW_SAMPLING_RATE, win_len)
            s2_mfccs[i, :] = mfcc(signal[s2_index_start:s2_index_end+1], NEW_SAMPLING_RATE, winlen=win_len, winfunc=np.hamming, nfft=nfft)[0]

        
        if not np.isnan(state_starts[3,i]) and not np.isnan(state_ends[3,i]):
            # Get index
            dia_index_start = int(state_starts[3,i] * NEW_SAMPLING_RATE)
            dia_index_end = int(state_ends[3,i] * NEW_SAMPLING_RATE)

            # Save median power spectrum
            for j in range(9):
                dia_med_power_spectrums[j].append(np.median(bands_power_spectrum[j][dia_index_start:dia_index_end]))
            
            # Save MFCC features
            win_len = state_ends[3,i] - state_starts[3,i] + 0.001
            nfft = calculate_nfft(NEW_SAMPLING_RATE, win_len)
            dia_mfccs[i, :] = mfcc(signal[dia_index_start:dia_index_end+1], NEW_SAMPLING_RATE, winlen=win_len, winfunc=np.hamming, nfft=nfft)[0]


    s1_pwd_features = np.array(([np.nanmean(x) for x in s1_med_power_spectrums]))
    sys_pwd_features = np.array(([np.nanmean(x) for x in sys_med_power_spectrums]))
    s2_pwd_features = np.array(([np.nanmean(x) for x in s2_med_power_spectrums]))
    dia_pwd_features = np.array(([np.nanmean(x) for x in dia_med_power_spectrums]))

    power_spectrum_features = np.hstack((s1_pwd_features, sys_pwd_features, s2_pwd_features, dia_pwd_features))
    # print(power_spectrum_features)

    s1_mfccs_features = np.nanmean(s1_mfccs, axis=0)
    sys_mfccs_features = np.nanmean(sys_mfccs, axis=0)
    s2_mfccs_features = np.nanmean(s2_mfccs, axis=0)
    dia_mfccs_features = np.nanmean(dia_mfccs, axis=0)

    mfcc_features = np.hstack((s1_mfccs_features, sys_mfccs_features, s2_mfccs_features, dia_mfccs_features))
    # print(mfcc_features)

    frequency_domain_features = np.hstack((power_spectrum_features, mfcc_features))
    # print(frequency_domain_features)

    if np.any(np.isnan(frequency_domain_features)):
        tqdm.write(f"Frequency: {frequency_domain_features}")

    return frequency_domain_features


# Testing
if __name__ == "__main__":
    data_folder = join(DATA_PATH, 'test')
    patient_files = find_patient_files(data_folder)

    for patient in patient_files:
        current_patient_data = load_patient_data(patient)
        recordings_paths, current_recordings = load_recordings(data_folder, current_patient_data, get_paths=True)
        
        features = get_PCG_features(current_recordings[0], recordings_paths[0])
        print(features)
        print(features.shape)
        
        break
