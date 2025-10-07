"""
Preprocess the Oldenburg dataset for the target speaker extraction scenario.
In this scenario, the target speech is the own voice.
An optional additive noise and an optional interfering talker are added.
"""
import pathlib
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess_vibravox_TSE import (
    compute_mean_std_indep,
    process_example,
    select_interferer,
)

# seed all randomness
base_seed = 123
random.seed(base_seed)
np.random.seed(base_seed)


def main(refmic_ch: int, sensor_ch: int, num_processes: int=32):
    
    # Define paths
    input_dir = pathlib.Path('/home/ubuntu/data/vibravox_wav')

    output_dir = pathlib.Path(f'/home/ubuntu/data/VV_TSE/') 
    output_dir.mkdir(exist_ok=True)

    subset_speech = 'speech_clean'
    subset_noise = 'speechless_noisy'
    
    # read sensor key from file
    key_list_txt = input_dir.joinpath('channel_indexing.txt')
    with open(key_list_txt, 'r') as f:
        key_list = f.readlines()
        sensor_key = key_list[sensor_ch].split(' ')[1].strip()
        #refmic_key = key_list[refmic_ch].split(' ')[1].strip()

    sensor_output_dir = output_dir.joinpath(sensor_key+f'_separate_eval_sets')
    sensor_output_dir.mkdir(exist_ok=True)

    fs_in = 16000
    fs_out = 16000

    example_duration_sec = 3.0

    snr_ranges = {
        'test_noise': [-10, 10],
        'test_interferer': [-10, 10],
        'test_both': [-10, 10],
        'test_clean': [-10, 10],
    }

    probabilities = {
        'test_noise': [1.0, 0.0],
        'test_interferer': [0.0, 1.0],
        'test_both': [1.0, 1.0],
        'test_clean': [0.0, 0.0],
    }

    # set up the split list
    split_list = [
        'test_noise', 'test_interferer', 'test_both', 'test_clean'
    ]

    # compute anti-clip factor
    add_db = 6 
    headroom_db = 10 # dB headroom 

    # compute factor based on absolute minimum SNR in all splits
    absolute_minimum_snr = min(min(snr) for snr in snr_ranges.values())
    anticlip_factor = 10 ** ((absolute_minimum_snr - add_db - headroom_db) / 20) 

    # for each split, load the dataset previously downloaded
    for split in split_list:

        print(f"Pre-processing {split} split")

        # get the first part (train/validation/test) of the split name
        actual_split = split.split('_')[0]
        csv_file = input_dir.joinpath(f'file_paths_{actual_split}_{subset_speech}.csv')
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
        noise_csv_file = input_dir.joinpath(f'file_paths_{actual_split}_{subset_noise}.csv')
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
    