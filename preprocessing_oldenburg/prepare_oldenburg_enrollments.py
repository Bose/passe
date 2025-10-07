"""
This script loops over the different dataset splits of the VIBRAVOX dataset and computes the embeddings for each split.
Each file gets its own embedding file.
The embeddings are saved in a parallel directory as the original files, with the same name but with a .npy extension.
The script uses the VIBRAVOX dataset and the VIBRAVOX model to compute the embeddings.
The script uses the ECAPA2 model to compute the embeddings for each file in the dataset.
"""
import pathlib
import random
from multiprocessing import Pool

import pandas as pd
import torch
import tqdm

# seed for reproducibility
random.seed(42)
torch.manual_seed(42)


def main():

    dataset_dir = pathlib.Path(
        '/home/ubuntu/data/TSE_oldenburg_preprocessed/'
        #'outer_inear_OL_interf'
        'outer_inear_separate_eval_sets_OL_interf'
    ) 

    # set up the split list
    #split_list = ['train', 'validation']

    split_list = [
        'test_noise', 'test_interferer', 'test_both', 'test_clean'
    ]

    # loop over splits
    for split in split_list:
        
        # load csv file with pandas
        csv_file = dataset_dir.joinpath(f'file_paths_{split}.csv')
        df = pd.read_csv(csv_file, dtype={'target': str, 'talker': str})

        output_csv_file = dataset_dir.joinpath(f'file_paths_{split}_embeddings.csv')

        # Prepare arguments for multiprocessing
        args = [(row['target'], row['talker']) for i, row in df.iterrows()] 

        # Preprocess and save audio files
        file_paths = []

        num_processes = 8

        # multiprocessing: process all examples in parallel (segment, resample, add noise, save audio file)
        with Pool(processes=num_processes) as pool:
            results = list(tqdm.tqdm(pool.imap(process_example, args), total=len(args)))
        file_paths.extend(results)

        # Save file paths to CSV
        df = pd.DataFrame(file_paths)
        df.to_csv(output_csv_file, index=True) 

        
def process_example(args):

    file_path_target, talker_id = args

    return {
        'audio_file': file_path_target,
        'talker': talker_id
    }


if __name__ == "__main__":
    main()