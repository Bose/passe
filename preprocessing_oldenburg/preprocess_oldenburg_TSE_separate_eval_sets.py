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

from preprocess_oldenburg_TSE import (
    compute_mean_std_indep,
    print_noise_stats,
    process_example,
)

# seed all randomness
base_seed = 123
random.seed(base_seed)
np.random.seed(base_seed)


def main(setup_name: str, reference_channels: list, sensor_channels: list, 
         num_processes: int=32, use_precleaned: bool=True):
    """
    Preprocess the Oldenburg dataset by segmenting, resampling to 16kHz, 
    and adding noise at a random SNR from a given range.

    This function does all the splits (train, validation, test) for
    one sensor and writes audio files, normalization values,
    and a CSV with a file list of corresponding noisy and target signals
    and the SNR the noisy signal was mixed at.

    All channels given in the reference and sensor channel lists are assumed to be paired,
    So for left and right reference (outer) mic the corresponding list element of the 
    sensor channel list should contain the same side device in-ear microphone index.
    In the output/preprocessed files, they are treated as more data of the same.
    Assumes that first index pair is left side, second is right side.

    Requires noise pre-generated with simulate_odenburg_noise.py

    Args:
        setup_name (str): name of sensor setup, used for saving files to a subfolder
        reference_channel (list): indices of reference / broadband microphone channels
        sensor_channel (list): indices of sensor / in-ear microphone channels

    """
    # Define paths
    if use_precleaned:
        input_dir = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/speech_precleaned/')
    else:
        input_dir = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/speech/')


    output_dir = pathlib.Path(f'/home/ubuntu/data/TSE_oldenburg_preprocessed/{setup_name}_separate_eval_sets_OL_interf') 
    print(f"Output dir: {output_dir}")
    
    output_dir.mkdir(exist_ok=True)

    # directories for the noise of the left (default) and right side
    noise_dir = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_noise/{setup_name}')
    noise_dir_right = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_noise/{setup_name}_right_side')

    # directories for the interfering speech of the left (default) and right side
    interferer_dir = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_interferer_OL/{setup_name}')
    interferer_dir_right = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_interferer_OL/{setup_name}_right_side')
    print(f"warning: using OL interferer, output dir: {interferer_dir} and {interferer_dir_right}")

    # exclude utterances that are not proper sentences 
    highest_utt = 306
    
    fs_out = 16000

    if use_precleaned:
        fs_in = 16000
    else:
        fs_in = 44100 

    example_duration_sec = 3.0

    snr_ranges = {
        'test_noise': [-10, 10],
        'test_interferer': [-10, 10],
        'test_both': [-10, 10],
        'test_clean': [-10, 10]
    }

    probabilities = {
        'test_noise': [1.0, 0.0],
        'test_interferer': [0.0, 1.0],
        'test_both': [1.0, 1.0],
        'test_clean': [0.0, 0.0],
    }

    old_talker_split = {
        'train': ['VP_01', 'VP_02', 'VP_04', 'VP_05', 
                  'VP_06', 'VP_10', 'VP_11', 'VP_13', 
                  'VP_14', 'VP_15', 
                    'VP_23', 'VP_24'],    
        'validation': ['VP_18', 'VP_21',],
        'test': ['VP_26', 'VP_27', 'VP_28', 'VP_29'],
    }

    talker_split = {
        'test_noise': old_talker_split['test'],
        'test_interferer': old_talker_split['test'],
        'test_both': old_talker_split['test'],
        'test_clean': old_talker_split['test'],
    }

    # set up the split list
    split_list = [
        'test_noise', 'test_interferer', 'test_both', 'test_clean'
    ]


    # compute anti-clip factor
    add_db = 0 
    add_db = 12 # 20*log10(3) dB
    print(f'warning: using add_db = {add_db} dB')
    headroom_db = 0 # dB headroom 

    # compute factor based on absolute minimum SNR in all splits
    absolute_minimum_snr = min(min(snr) for snr in snr_ranges.values())
    anticlip_factor = 10 ** ((absolute_minimum_snr - add_db - headroom_db) / 20) 

    # for each split, load the dataset previously downloaded
    for split in split_list:

        # set up speech dataset file_list for this split
        speech_file_list = []
        for talker_str in talker_split[split]:
            talker_files = input_dir.joinpath(talker_str).iterdir()

            # exclude vowel sounds and silence example (numbers 306 and above)
            talker_files = [fname for fname in talker_files if int(fname.stem.split('_')[-1]) < highest_utt]

            speech_file_list.extend(talker_files) 

        csv_file = output_dir.joinpath(f'file_paths_{split}.csv')
        normalization_file = output_dir.joinpath(f'normalization_file_{split}.npz')
        means = []
        stds = []

        output_split_dir = output_dir.joinpath(split)
        output_split_dir.mkdir(exist_ok=True)

        snr_range_db = snr_ranges[split]

        noise_prob, interferer_prob = probabilities[split]

        # Preprocess and save audio files
        file_paths = []

        # Process both side device recordings
        for side_idx, side in enumerate(['left', 'right']):

            print(f"Pre-processing {split} split, {side} side")

            # select the channels for this side device
            reference_channel = reference_channels[side_idx]
            sensor_channel = sensor_channels[side_idx]

            # Compute mean and std for splits
            split_side_mean, split_side_std = compute_mean_std_indep(
                file_list=speech_file_list, reference_channel=reference_channel, 
                sensor_channel=sensor_channel, anticlip_factor=anticlip_factor)
        
            # scale sensor audio to match reference microphone audio
            sensor_scale_factor = (split_side_std[0] / split_side_std[1]) 

            if np.isnan(sensor_scale_factor).any() or np.isinf(sensor_scale_factor).any():
                print(f"sensor scale factor is NaN or Inf: {sensor_scale_factor}")
                raise ValueError

            # save the statistics to a file (since sensor is scaled, only need reference stats)
            split_side_mean = split_side_mean[0]
            split_side_std=split_side_std[0]
                
            means.append(split_side_mean)
            stds.append(split_side_std)

            # set up noise and interferer datasets
            if side == 'right':
                side_noise_dir = noise_dir_right
                side_interferer_dir = interferer_dir_right
            else:
                side_noise_dir = noise_dir
                side_interferer_dir = interferer_dir

            noise_file_dict = {}
            interferer_file_dict = {}
            for talker_str in talker_split[split]:
                noise_file_dict[talker_str] = list(side_noise_dir.joinpath(talker_str).iterdir())
                interferer_file_dict[talker_str] = list(side_interferer_dir.joinpath(talker_str).iterdir())


            # fill argument list 
            args = []
            for i, example in enumerate(speech_file_list):

                talker_str = example.parent.name

                # select random noise file
                rnd_noise_idx = np.random.randint(low=0, high=len(noise_file_dict[talker_str]))
                random_noise = noise_file_dict[talker_str][rnd_noise_idx]

                # select random interferer file
                rnd_interferer_idx = np.random.randint(low=0, high=len(interferer_file_dict[talker_str]))
                random_interferer = interferer_file_dict[talker_str][rnd_interferer_idx]

                args.append((example, i, random_noise, random_interferer, 
                             noise_prob, interferer_prob, fs_in, fs_out,
                             example_duration_sec, output_split_dir, reference_channel, sensor_channel, 
                             snr_range_db, anticlip_factor, sensor_scale_factor))

            # multiprocessing: process all examples in parallel (segment, resample, add noise, save audio file)
            with Pool(processes=num_processes, initializer=np.random.seed) as pool:
                results = list(tqdm(pool.imap(process_example, args), total=len(args)))
        
            file_paths.extend(results)

        # Save file paths to CSV
        df = pd.DataFrame(file_paths)
        df.to_csv(csv_file, index=True)

        # average left and right side means and stds
        split_mean = np.mean(means)
        split_std = np.mean(stds)

        # save the statistics to a file
        np.savez(normalization_file, mean=split_mean, std=split_std)

        # print the statistics
        print_noise_stats(csv_file)


if __name__ == '__main__':

    num_processes = 16

    # 0: Lavalier-microphone clipped to the shirt neck, shirt collar etc. of the talker
    # 1: Reference microphone about 50 cm in front of the talker
    # 2: Left in-ear microphone Hearpiece
    # 3: Left concha microphone Hearpiece
    # 4: Right in-ear microphone Hearpiece
    # 5: Right concha microphone Hearpiece

    setup_choices = {
        'closetalk_inear': {
            'reference_channels': [1, 1], 
            'sensor_channels': [2, 4]
        }, 
        'outer_inear': {
            'reference_channels': [3, 5],
            'sensor_channels': [2, 4]
        }
        # add other combinations here if desired
    }
    
    # SELECT HERE
    setup_name = 'outer_inear'

    use_OL_interferer = True
    print(f'using setup {setup_name} with OL interferer= {use_OL_interferer}')

    indep_norm = True
    use_precleaned = True

    reference_channels = setup_choices[setup_name]['reference_channels']
    sensor_channels = setup_choices[setup_name]['sensor_channels']

    main(setup_name, reference_channels=reference_channels, sensor_channels=sensor_channels,
         num_processes=num_processes, indep_norm=indep_norm, use_precleaned=use_precleaned,
         use_OL_interferer=use_OL_interferer)
    