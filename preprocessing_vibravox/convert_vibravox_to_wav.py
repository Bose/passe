import pathlib
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm

# seed all randomness
base_seed = 123
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed) 



def main(num_processes: int=8):
    # Define paths
    input_dir = pathlib.Path('/home/ubuntu/data/vibravox')
    output_dir = pathlib.Path('/home/ubuntu/data/vibravox_wav') 
        
    output_dir.mkdir(exist_ok=True)

    subset_speech = 'speech_clean'
    subset_noise = 'speechless_noisy'

    key_list = [
        'audio.headset_microphone',
        'audio.soft_in_ear_microphone',
        'audio.rigid_in_ear_microphone',
        #'audio.forehead_accelerometer', 
        #'audio.temple_vibration_pickup',
        #'audio.throat_microphone'
    ]

    # write the numbered key list to a file
    key_list_txt = output_dir.joinpath('channel_indexing.txt')
    with open(key_list_txt, 'w') as f:
        for i, key in enumerate(key_list):
            line = f"{i} {key}\n"
            print(line)
            f.write(line)

    fs_in = 48000 
    fs_out = 16000
    resample_transform = torchaudio.transforms.Resample(orig_freq=fs_in, new_freq=fs_out)

    # for each split, load the dataset previously downloaded
    for split in ['train', 'validation', 'test']: 
        for subset in [subset_speech, subset_noise]:
            print(f"Pre-processing {subset} {split} split")

            dataset_speech = load_from_disk(input_dir.joinpath(subset).joinpath(split))

            output_split_dir = output_dir.joinpath(split)
            output_split_dir.mkdir(exist_ok=True)

            csv_file = output_dir.joinpath(f'file_paths_{split}_{subset}.csv')

            # Prepare arguments for multiprocessing
            args = [(example, i, fs_out, output_split_dir, key_list, subset, resample_transform
                    ) for i, example in enumerate(dataset_speech)] 

            file_paths = []

            # multiprocessing: process all examples in parallel (append channels, resample, save as wav file)
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(process_example, args), total=len(args)))

            file_paths.extend(results)
            
            # Save file paths to CSV
            df = pd.DataFrame(file_paths)
            df.to_csv(csv_file, index=True) 


def process_example(args):

    example, i, fs_out, output_split_dir, key_list, subset, resample_transform = args

    # Load audio for each key in key_list
    audio = []
    for key in key_list:
        audio.append(example[key]["array"])

    # Concatenate all channels
    audio = torch.tensor(np.stack(audio, axis=0), dtype=torch.float32)

    # Resample
    audio = resample_transform(audio)

    # read speaker id
    speaker_id = example['speaker_id']

    # save audio file
    output_file = output_split_dir.joinpath(f'{subset}_{i}.wav')
    try:
        torchaudio.save(output_file, audio, fs_out) 
    except Exception as e:
        print(f"Error writing files: {e}")
    
    # return table entry containing filename and speaker id
    return {'filename': output_file, 'speaker_id': speaker_id}


if __name__ == '__main__':
    main(num_processes=8)