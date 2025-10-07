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
import scipy.signal as sig
import soundfile as sf
from tqdm import tqdm

from preprocess_vibravox import select_random_segment

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

    output_dir = pathlib.Path(f'/home/ubuntu/data/TSE_oldenburg_preprocessed/{setup_name}_OL_interf') 
    print(f"output dir: {output_dir}")

    output_dir.mkdir(exist_ok=True)

    # directories for the noise of the left (default) and right side
    noise_dir = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_noise/{setup_name}')
    noise_dir_right = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_noise/{setup_name}_right_side')

    # directories for the interfering speech of the left (default) and right side
    interferer_dir = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_interferer_OL/{setup_name}')
    interferer_dir_right = pathlib.Path(f'/home/ubuntu/data/hearpiece_ownvoice/simulated_interferer_OL/{setup_name}_right_side')
    
    # exclude utterances that are not proper sentences 
    highest_utt = 306
    
    fs_out = 16000

    if use_precleaned:
        fs_in = 16000
    else:
        fs_in = 44100 

    example_duration_sec = 3.0

    snr_ranges = {
        'train': [-10, 10],
        'validation': [-10, 10],
    }

    probabilities = {
        'train': [0.75, 0.75],
        'validation': [0.75, 0.75],
    }

    talker_split = {
        'train': ['VP_01', 'VP_02', 'VP_04', 'VP_05', 
                  'VP_06', 'VP_10', 'VP_11', 'VP_13', 
                  'VP_14', 'VP_15', 
                    'VP_23', 'VP_24'],   
        'validation': ['VP_18', 'VP_21',], 
        #'test': ['VP_26', 'VP_27', 'VP_28', 'VP_29'],
    }
    

    # compute anti-clip factor
    add_db = 0 
    add_db = 12 # 20*log10(3) dB
    print(f'warning: using add_db = {add_db} dB')

    headroom_db = 0 # dB headroom 

    # compute factor based on absolute minimum SNR in all splits
    absolute_minimum_snr = min(min(snr) for snr in snr_ranges.values())
    anticlip_factor = 10 ** ((absolute_minimum_snr - add_db - headroom_db) / 20) 

    # for each split, load the dataset previously downloaded
    for split in ['train', 'validation']:

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
        # -> they can be a bit misleading here since some entries are 999, 
        # so the averages are oddly high and not the actual average SNRs 
        #print_noise_stats(csv_file) 


def print_noise_stats(csv_file: str):
    """
    Compute the mean and std of the SNR, SIR, SINR in the dataset and print them
    """
    df = pd.read_csv(csv_file)

    # compute mean and std of SNR, SIR, SINR
    snr_mean = df['snr_db'].mean()
    sir_mean = df['sir_db'].mean()
    sinr_mean = df['sinr_db'].mean()

    print(f'SNR: {snr_mean:.2f} dB')
    print(f'SIR: {sir_mean:.2f} dB')
    print(f'SINR: {sinr_mean:.2f} dB')
 


def process_example(args):
    """
    Function to process a single example from the dataset.

    Args:
        args (tuple): Tuple containing the following elements:
            example (str or path): Example (filename) from the dataset.
            i (int): Index of the example.
            noise_example (str or path): Random noise filename.
            interferer_example (str or path): Random interfering speech filename.
            fs_in (int): Input sampling rate.
            fs_out (int): Output sampling rate.
            example_duration_sec (float): Duration of the output audio.
            output_split_dir (pathlib.Path): Path to the output directory.
            refmic_key (str): Key to access the reference microphone audio in the dataset.
            sensor_key (str): Key to access the sensor audio in the dataset.
            snr_range_db (list): Range of SNR values to use for adding noise.
            anticlip_factor (float): scaling factor for preventing clipping when saving to wav.
            sensor_scale_factor (float): scaling factor to match sensor microphone level
                to reference microphone level.
    Returns:
        dict: Dictionary containing the file paths of the target and noisy audio files.
    """
    (example, i, 
     noise_example, interferer_example, 
     noise_prob, interferer_prob,
     fs_in, fs_out, 
     example_duration_sec, 
     output_split_dir, 
     reference_channel, sensor_channel, 
     snr_range_db, anticlip_factor, sensor_scale_factor) = args
    
    # seed all randomness for this process
    random.seed(base_seed + i)
    np.random.seed(base_seed + i)

    # read audio and select channels
    audio, _ = sf.read(example, always_2d=True)
    
    mc_speech = audio[:, [reference_channel, sensor_channel]] # resulting shape (samples, 2)
    mc_speech[:, 1] *= sensor_scale_factor

    talker_str = example.parent.name

    noisy, target, clean_inear, snr_db, sir_db, sinr_db = preprocess_audio(
        mc_speech=mc_speech, fs_in=fs_in, fs_out=fs_out,
        example_duration_sec=example_duration_sec,
        noise_example=noise_example,
        interferer_example=interferer_example,
        noise_prob=noise_prob, interferer_prob=interferer_prob,
        snr_range_db=snr_range_db,
        anticlip_factor=anticlip_factor,
        sensor_scale_factor=sensor_scale_factor)
    
    target_file = output_split_dir.joinpath(f'target_{i}.wav')
    clean_inear_file = output_split_dir.joinpath(f'clean_inear_{i}.wav')
    noisy_file = output_split_dir.joinpath(f'noisy_{i}.wav')

    try:
        sf.write(target_file, target, fs_out) 
        sf.write(clean_inear_file, clean_inear, fs_out)
        sf.write(noisy_file, noisy, fs_out)
    except Exception as e:
        print(f"Error writing files: {e}")
    
    return {'target': target_file, 
            'clean_inear': clean_inear_file,
            'noisy': noisy_file, 
            'snr_db': snr_db, 
            'sir_db': sir_db, 
            'sinr_db': sinr_db,
            'talker': talker_str}


def preprocess_audio(mc_speech: np.ndarray, fs_in: int, fs_out: int,
                     noise_example: pathlib.Path, interferer_example: pathlib.Path, 
                     noise_prob: float=0.75, interferer_prob: float=0.75,
                     example_duration_sec: float=3.0, snr_range_db=[-10, 10],
                     anticlip_factor: float=1.0, sensor_scale_factor: float=1.0
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Function to preprocess audio (segmenting, resampling, adding noise)

    Args:
        mc_speech (np.ndarray): Multi-channel audio signal of shape (samples, channels).
                                Here, it is assumed that the first channel is the target/reference.
        fs_in (int): Input sampling rate.
        noise_example (pathlib.Path): Path to the noise example.
        interferer_example (pathlib.Path): Path to the interfering speech example.
        example_duration_sec (float): Duration of the output audio.
        snr_range_db (list): Range of SNR values to use for adding noise.
        anticlip_factor (float): scaling factor for preventing clipping.
        sensor_scale_factor (float): scaling factor to match sensor microphone level

    Returns:
        audio_noisy (np.ndarray): Noisy audio signal.
        audio (np.ndarray): Clean audio signal.
        snr_db (float): signal-to-noise-ratio in decibel used for mixing speech and noise.
    """
    target_ch_idx = 0
    inear_ch_idx = 1
    
    # Segment
    mc_speech = select_random_segment(mc_speech, example_duration_sec, fs_in) 

    # Resample
    mc_speech = sig.resample_poly(up=fs_out, down=fs_in, x=mc_speech, axis=0)

    rms_speech = np.sqrt(np.mean(mc_speech[:, target_ch_idx]**2))

    target = mc_speech[:, [target_ch_idx]]

    # make sure noise and interferer are not None
    if noise_example is None or interferer_example is None:
        raise ValueError("noise or interferer example is None")

    # randomly select with 75% chance if there is noise or not, and 50% chance if there is an interferer
    noise_present = np.random.choice([True, False], p=[noise_prob, 1.0-noise_prob])
    interferer_present = np.random.choice([True, False], p=[interferer_prob, 1.0-interferer_prob])


    if noise_present:
        # read noise and interferer examples (already resampled to 16 kHz in simulation)
        mc_noise, _ = sf.read(noise_example, always_2d=True) # shape (samples, 2)
        mc_noise[:, 1] *= sensor_scale_factor

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
        mc_interferer, _ = sf.read(interferer_example, always_2d=True) # shape (samples, 2)
        mc_interferer[:, 1] *= sensor_scale_factor

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
        sinr = 999

    else: # noisy speech
        if noise_present and interferer_present: # noise and interferer
            audio_disturbances = scaling_factor_noise * mc_noise + scaling_factor_interferer * mc_interferer
        
        elif noise_present and not interferer_present: # only noise
            audio_disturbances = scaling_factor_noise * mc_noise

        elif not noise_present and interferer_present: # only interferer present
            audio_disturbances = scaling_factor_interferer * mc_interferer

        audio_noisy = anticlip_factor * (mc_speech + audio_disturbances)

        rms_noise_est = np.sqrt(np.mean(audio_disturbances[:, target_ch_idx]**2))
        sinr = 20*np.log10(rms_speech / rms_noise_est)


    # for signals without noise or intererer presence, re-compute the SINR and throw warning if it's below 20 dB
    if not noise_present and not interferer_present:
        audio_disturbances_est = audio_noisy/anticlip_factor - mc_speech
        rms_noise_est = np.sqrt(np.mean(audio_disturbances_est[:, target_ch_idx]**2))
        sinr_est = 20*np.log10(rms_speech / rms_noise_est)
        if sinr_est < 20:
            print(f"Warning: SINR for clean file below 20 dB, estimated SINR: {sinr_est}")

    audio_target = anticlip_factor * target
    clean_inear = anticlip_factor * mc_speech[:, [inear_ch_idx]]

    # if the noisy audio will clip or has invalid values, print a warning
    if np.amax(np.abs(audio_noisy)) > 1.0 :
        print(f"Warning: Noisy audio will clip, max speech: {np.amax(np.abs(mc_speech))}, noise rms: {np.amax(np.abs(mc_noise))}")
        raise ValueError

    if np.isnan(audio_noisy).any() or np.isinf(audio_noisy).any():
        print(f"warning: noisy audio is NaN or Inf")
        print(f"noise file: {noise_example}, has rms: {rms_noise} and factor {scaling_factor_noise}")
        print(f"interferer file: {interferer_example}, has rms: {rms_interferer} and factor {scaling_factor_interferer}")
        print(f"speech rms: {rms_speech}, shape {mc_speech.shape}")
        print(f"isnan: {np.isnan(audio_noisy).any()}, isinf: {np.isinf(audio_noisy).any()}")
        print(f"isnan speech: {np.isnan(mc_speech).any()}, isinf speech: {np.isinf(mc_speech).any()}")
        raise ValueError

    return audio_noisy, audio_target, clean_inear, snr_db_desired, sir_db_desired, sinr


def compute_mean_std_indep(file_list: list, reference_channel: int,
                           sensor_channel: int, 
                           anticlip_factor: float=1.0) -> tuple[float, float]:
    """
    Function to compute the mean and standard deviation of the audio dataset.
    Assumes the audio is stored in the dataset as a numpy array.
    
    Args:
        dataset (datasets.Dataset): Dataset containing audio files.
        reference_channel (int): index of the reference channel to compute 
                                statistics for
        sensor_channel (int): index of the sensor channel to compute 
                                statistics for
        anticlip_factor (float): scaling factor to prevent clipping, needs to be 
                                accounted for here as well.
        
    Returns:
        float: Mean of the audio dataset.
        float: Standard deviation of the audio dataset.
    """
    total_sums = np.array([0., 0.])
    total_squared_sums = np.array([0., 0.])
    total_count = 0

    for filename in file_list:
        audio, _ = sf.read(filename, always_2d=True)
        audio_selected = anticlip_factor * audio[:, [reference_channel, sensor_channel]]

        for c in range(2): # iterate over channels
            total_sums[c] += np.sum(audio_selected[:, c])
            total_squared_sums[c] += np.sum(audio_selected[:, c]**2)

        total_count += audio_selected.shape[0]

    # https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    total_mean = total_sums / total_count
    total_std = np.sqrt(total_count * total_squared_sums - total_sums**2) / total_count
    
    return total_mean, total_std


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

    use_precleaned = True

    reference_channels = setup_choices[setup_name]['reference_channels']
    sensor_channels = setup_choices[setup_name]['sensor_channels']

    main(setup_name, reference_channels=reference_channels, sensor_channels=sensor_channels,
         num_processes=num_processes, use_precleaned=use_precleaned)