"""
Preprocess the Vibravox dataset for the own voice+noise+interferer scenario.
This script assumes that the dataset is already downloaded and converted to wav.
"""
import pathlib
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

# seed all randomness
base_seed = 123
random.seed(base_seed)
np.random.seed(base_seed)

# Add interferer columns to combined_df
def select_interferer(row, file_paths_df):
    # Filter out rows with the same speaker_id
    filtered_df = file_paths_df[file_paths_df['speaker_id'] != row['speech_speaker_id']]
    # Randomly select one row
    selected_row = filtered_df.sample(n=1, random_state=base_seed + row.name).iloc[0]
    return selected_row['filename'], selected_row['speaker_id']


def select_random_segment(audio: np.ndarray, segment_length_sec: float, fs: int) -> np.ndarray:
    """
    Function to select a random segment of audio of a given length.
    If the audio is shorter than the segment length, it is padded.
    Assumes the audio has shape (samples, channels).

    Args:
        audio (numpy.ndarray): Audio signal of shape (samples, channels).
        segment_length_sec (float): Length of the segment in seconds.
        fs (int): Sampling rate of the audio.

    Returns:
        numpy.ndarray: Random segment of audio.
    """
    num_samples = audio.shape[0]
    segment_samples = int(segment_length_sec * fs)
    
    if num_samples < segment_samples:
        # If the audio is shorter than the segment length, pad it
        pad_amount = segment_samples - num_samples
        padded_audio = np.pad(audio, ((pad_amount, 0), (0,0)), mode='constant')

        # roll padded audio by random amount
        roll_shift = np.random.randint(0, pad_amount)
        return np.roll(padded_audio, shift=roll_shift, axis=0)
    
    elif num_samples > segment_samples:
        
        # Randomly select a starting point
        start_idx = np.random.randint(0, num_samples - segment_samples)
        end_idx = start_idx + segment_samples
    
        return audio[start_idx:end_idx, :]
    
    else: # in case audio has already exactly the desired length:
        # do nothing
        return audio
    

def compute_mean_std_indep(file_paths_df, refmic_ch: int, sensor_ch: str,  
                     anticlip_factor: float=1.0):
    """
    Function to compute the mean and standard deviation of the audio dataset.
    Assumes the audio is stored in the dataset as a numpy array.
    
    Args:
        dataset (datasets.Dataset): Dataset containing audio files.
        refmic_ch (int): Channel index to access the reference microphone audio.
        sensor_ch (int): Channel index to access the sensor audio.
        anticlip_factor (float): scaling factor to prevent clipping, needs to be 
                                accounted for here as well.
        
    Returns:
        float: Mean of the audio dataset.
        float: Standard deviation of the audio dataset.
    """
    total_sums = np.array([0., 0.])
    total_squared_sums = np.array([0., 0.])
    total_count = 0

    for _, row in file_paths_df.iterrows():
        mc_audio, _ = sf.read(row['filename'], always_2d=True)
        audios = [
            anticlip_factor * mc_audio[:, refmic_ch],
            anticlip_factor * mc_audio[:, sensor_ch]
        ]

        for c in range(2): # iterate over channels
            
            total_sums[c] += np.sum(audios[c])
            total_squared_sums[c] += np.sum(audios[c]**2)

        total_count += audios[0].size

    # https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    total_mean = total_sums / total_count
    total_std = np.sqrt(total_count * total_squared_sums - total_sums**2) / total_count
    
    return total_mean, total_std



def main(sensor_ch: int, refmic_ch: int, num_processes: int=32):
    """
    Preprocess the Vibravox dataset by segmenting, resampling to 16kHz, 
    and adding noise and interferer at a random SNR from a given range.

    This function does all the splits (train, validation, test) for
    one sensor and writes audio files, normalization values,
    and a CSV with a file list of corresponding noisy and target signals
    and the SNR the noisy signal was mixed at.

    Args:
        sensor_ch (int): the channel index of the sensor in the dataset.
        refmic_ch (int): the channel index of the reference microphone in the dataset.
        num_processes (int): number of processes for multiprocessing.Pool().
    """

    print(f"Preprocessing sensor ch {sensor_ch} and refmic ch {refmic_ch}")
    
    # Define paths
    input_dir = pathlib.Path('/home/ubuntu/data/vibravox_wav')
    output_dir = pathlib.Path('/home/ubuntu/data/VV_TSE') 

    output_dir.mkdir(exist_ok=True)

    subset_speech = 'speech_clean'
    subset_noise = 'speechless_noisy'

    # read sensor key from file
    key_list_txt = input_dir.joinpath('channel_indexing.txt')
    with open(key_list_txt, 'r') as f:
        key_list = f.readlines()
        sensor_key = key_list[sensor_ch].split(' ')[1].strip()
        refmic_key = key_list[refmic_ch].split(' ')[1].strip()

    sensor_output_dir = output_dir.joinpath(sensor_key)
    sensor_output_dir.mkdir(exist_ok=True)

    fs_out = 16000
    fs_in = 16000 

    example_duration_sec = 3.0

    snr_ranges = {
        'train': [-10, 10],
        'validation': [-10, 10]
    }

    probabilities = {
        'train': [0.75, 0.75],
        'validation': [0.75, 0.75]
    }

    # compute anti-clip factor
    add_db = 6
    headroom_db = 10 # dB headroom

    # compute factor based on absolute minimum SNR in all splits
    absolute_minimum_snr = min(min(snr) for snr in snr_ranges.values())
    anticlip_factor = 10 ** ((absolute_minimum_snr - add_db - headroom_db) / 20) 

    # for each split, load the dataset previously downloaded
    for split in ['train', 'validation']: 
        print(f"Pre-processing {subset_speech} {split} split")

        csv_file = input_dir.joinpath(f'file_paths_{split}_{subset_speech}.csv')
        file_paths_df = pd.read_csv(csv_file)

        output_csv_file = sensor_output_dir.joinpath(f'file_paths_{split}.csv')

        output_split_dir = sensor_output_dir.joinpath(split)
        output_split_dir.mkdir(exist_ok=True)

        normalization_file = sensor_output_dir.joinpath(f'normalization_file_{split}.npz')

        # Compute mean and std for splits
        split_mean, split_std = compute_mean_std_indep(
            file_paths_df, refmic_ch=refmic_ch, sensor_ch=sensor_ch,
            anticlip_factor=anticlip_factor)
        
        # scale sensor audio to match reference microphone audio
        sensor_scale_factor = (split_std[0] / split_std[1]) 

        # save the statistics to a file
        np.savez(normalization_file, mean=split_mean[0], std=split_std[0])

        # Load noise dataset
        noise_csv_file = input_dir.joinpath(f'file_paths_{split}_{subset_noise}.csv')
        noise_file_paths_df = pd.read_csv(noise_csv_file)

        snr_range_db = snr_ranges[split]
        noise_prob, interferer_prob = probabilities[split]

        # create a dataframe with randomly selected rows of the noise file paths dataframe
        noise_file_paths_sampled_df = noise_file_paths_df.sample(
            n=len(file_paths_df), replace=True, random_state=base_seed
        ).reset_index(drop=True)

        # Drop unnecessary columns
        noise_file_paths_sampled_df = noise_file_paths_sampled_df[['filename']]
        file_paths_df.drop(columns=['Unnamed: 0'], inplace=True)

        # Combine the speech and noise dataframes
        combined_df = pd.concat(
            [file_paths_df.reset_index(drop=True), noise_file_paths_sampled_df],
            axis=1, keys=['speech', 'noise'])
        
        # Flatten the multi-level columns for easier manipulation
        combined_df.columns = ['_'.join(col).strip() for col in combined_df.columns.values]

        # Select a random interferer from the same split
        # and add it to the combined dataframe
        combined_df[['interferer_filename', 'interferer_speaker_id']] = combined_df.apply(
            lambda row: pd.Series(select_interferer(row, file_paths_df)), axis=1
        )

        # Prepare arguments for multiprocessing
        args = [(row, base_seed, i, 
                 noise_prob, interferer_prob,
                 fs_in, fs_out, 
                 example_duration_sec, output_split_dir, refmic_ch, sensor_ch, 
                 snr_range_db, anticlip_factor, sensor_scale_factor
                 ) for i, row in combined_df.iterrows()] 

        # Preprocess and save audio files
        file_paths = []

        # multiprocessing: process all examples in parallel (segment, resample, add noise, save audio file)
        with Pool(processes=num_processes, initializer=np.random.seed) as pool:
            results = list(tqdm(pool.imap(process_example, args), total=len(args)))

        file_paths.extend(results)
        
        # Save file paths to CSV
        df = pd.DataFrame(file_paths)
        df.to_csv(output_csv_file, index=True) 


def process_example(args):
    """
    Function to process a single example from the dataset.

    Args:
        args (tuple): Tuple containing the following elements:
            speech_row (dict): Example from the speech file path dataframe.
            noise_row (dict): Example from the noise file path dataframe.
            i (int): Index of the example.
            fs_in (int): Input sampling rate.
            fs_out (int): Output sampling rate.
            example_duration_sec (float): Duration of the output audio.
            output_split_dir (pathlib.Path): Path to the output directory.
            refmic_ch (int): Channel index to access the reference microphone audio.
            sensor_ch (int): Channel index to access the sensor audio.
            snr_range_db (list): Range of SNR values to use for adding noise.
            anticlip_factor (float): scaling factor for preventing clipping when saving to wav.
            sensor_scale_factor (float): scaling factor to match sensor microphone level
                to reference microphone level.
    Returns:
        dict: Dictionary containing the file paths of the target and noisy audio files.
    """
    (row, base_seed, i, noise_prob, interferer_prob, fs_in, fs_out, example_duration_sec, 
     output_split_dir, refmic_ch, sensor_ch, snr_range_db, anticlip_factor,
     sensor_scale_factor) = args
        
    # seed all randomness for this process
    random.seed(base_seed + i)
    np.random.seed(base_seed + i)

    speech_path = row['speech_filename']
    temp_mc_speech, _ = sf.read(speech_path, always_2d=True)
    refmic_speech = temp_mc_speech[:, refmic_ch]
    sensor_speech = sensor_scale_factor * temp_mc_speech[:, sensor_ch]
    mc_speech = np.vstack([refmic_speech, sensor_speech]).T # shape (2, samples).T = (samples, 2)

    target_ch_idx = 0
        
    # Segment
    mc_speech = select_random_segment(mc_speech, example_duration_sec, fs_in)
    target = mc_speech[:, [target_ch_idx]]
    rms_speech = np.sqrt(np.mean(target**2))

    # randomly select with 75% chance if there is noise or not, and 50% chance if there is an interferer
    noise_present = np.random.choice([True, False], p=[noise_prob, 1.0-noise_prob])
    interferer_present = np.random.choice([True, False], p=[interferer_prob, 1.0-interferer_prob])

    
    if noise_present:
        # read noise and interferer examples (already resampled to 16 kHz in simulation)
        temp_mc_noise, _ = sf.read(row['noise_filename'], always_2d=True) # shape (samples, 2)
        refmic_noise = temp_mc_noise[:, refmic_ch]
        sensor_noise = sensor_scale_factor * temp_mc_noise[:, sensor_ch]
        mc_noise = np.vstack([refmic_noise, sensor_noise]).T # shape (2, samples).T = (samples, 2)
        
        # segment
        mc_noise = select_random_segment(mc_noise, example_duration_sec, fs_out) 
        rms_noise = np.sqrt(np.mean(mc_noise[:, target_ch_idx]**2))

        # if noise segment is empty for some reason, noise is not present
        if rms_noise == 0.0:
            noise_present = False

    # check again - noise present?
    if noise_present:

        # mix to desired (random) SNR 
        snr_db_desired = np.random.uniform(snr_range_db[0], snr_range_db[1])
        snr_db_current = 20*np.log10(rms_speech / rms_noise)
        scaling_factor_noise = 10 ** ((snr_db_current - snr_db_desired) / 20)
    
    else: # noise absent
        snr_db_desired = 999
        scaling_factor_noise = 0.0

    if interferer_present:
        temp_mc_interferer, _ = sf.read(row['interferer_filename'], always_2d=True) # shape (samples, 2)
        refmic_interferer = temp_mc_interferer[:, refmic_ch]
        sensor_interferer = sensor_scale_factor * temp_mc_interferer[:, sensor_ch]
        mc_interferer = np.vstack([refmic_interferer, sensor_interferer]).T # shape (2, samples).T = (samples, 2)

        # segment
        mc_interferer = select_random_segment(mc_interferer, example_duration_sec, fs_out)
        rms_interferer = np.sqrt(np.mean(mc_interferer[:, target_ch_idx]**2))

        # if interferer segment is empty for some reason, interferer is not present
        if rms_interferer == 0.0 :
            interferer_present = False

    # check again - interferer present?
    if interferer_present:

        # mix to desired (random) SNR 
        sir_db_desired = np.random.uniform(snr_range_db[0], snr_range_db[1])
        sir_db_current = 20*np.log10(rms_speech / rms_interferer)
        scaling_factor_interferer = 10 ** ((sir_db_current - sir_db_desired) / 20)
    else:
        # noisy = clean
        sir_db_desired = 999
        scaling_factor_interferer = 0.0
    

    # mix the signals
    if not noise_present and not interferer_present: # clean speech
        audio_noisy = anticlip_factor * mc_speech
        sinr_db = 999

    else: # noisy speech
        if noise_present and interferer_present: # noise and interferer
            audio_disturbances = scaling_factor_noise * mc_noise + scaling_factor_interferer * mc_interferer
        
        elif noise_present and not interferer_present: # only noise
            audio_disturbances = scaling_factor_noise * mc_noise

        elif not noise_present and interferer_present: # only interferer present
            audio_disturbances = scaling_factor_interferer * mc_interferer

        audio_noisy = anticlip_factor * (mc_speech + audio_disturbances)

        rms_noise_est = np.sqrt(np.mean(audio_disturbances[:, target_ch_idx]**2))
        sinr_db = 20*np.log10(rms_speech / rms_noise_est)


    # for signals without noise or intererer presence, re-compute the SINR and throw warning if it's below 20 dB
    if not noise_present and not interferer_present:
        audio_disturbances_est = audio_noisy/anticlip_factor - mc_speech
        rms_noise_est = np.sqrt(np.mean(audio_disturbances_est[:, target_ch_idx]**2))
        sinr_est = 20*np.log10(rms_speech / rms_noise_est)
        if sinr_est < 20:
            print(f"Warning: SINR for clean file below 20 dB, estimated SINR: {sinr_est}")


    audio_target = anticlip_factor * target

    mc_speech = anticlip_factor * mc_speech

    # if the noisy audio will clip, print a warning
    if np.amax(np.abs(audio_noisy)) > 1.0 :
        print(f"Warning: Noisy audio will clip, max speech: {np.amax(np.abs(mc_speech))}, noise rms: {np.amax(np.abs(mc_noise))}")
     
    target_file = output_split_dir.joinpath(f'target_{i}.wav')
    noisy_file = output_split_dir.joinpath(f'noisy_{i}.wav')
    clean_inear_file = output_split_dir.joinpath(f'clean_inear_{i}.wav')

    try:
        sf.write(target_file, audio_target, fs_out)
        sf.write(noisy_file, audio_noisy, fs_out)
        sf.write(clean_inear_file, mc_speech[:, [1]], fs_out, subtype='PCM_32')
    except Exception as e:
        print(f"Error writing files: {e}")
    #print("skipping writing files")

    speaker_id = row['speech_speaker_id']
    
    return {'target': target_file, 
            'noisy': noisy_file, 
            'clean_inear': clean_inear_file, 
            'snr_db': snr_db_desired,
            'sir_db': sir_db_desired,
            'sinr_db': sinr_db,
            'talker': speaker_id}


if __name__ == '__main__':

    num_processes = 16

    #key_list = [
    #    'audio.headset_microphone',
    #    'audio.soft_in_ear_microphone',
    #    'audio.rigid_in_ear_microphone'
    #]
    
    # -> can look the index up in the txt file

    refmic_ch = 0
    #sensor_ch = 2
    sensor_ch = 1

    main(sensor_ch=sensor_ch, refmic_ch=refmic_ch, num_processes=num_processes)