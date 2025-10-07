import pathlib

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

base_seed = 123


class PreprocessedDataset(Dataset):
    """
    Torch dataset that manages preprocessed audio files from the preprocessed datasets.
    This assumes the dataset was preprocessed and saved as a CSV file with columns 'target' and 'noisy',
    as well as .wav files and corresponding dataset normalisation scaling values.

    Args:
        dataset_csv_path (pathlib.Path): Path to the CSV file containing the dataset file list.
    """

    def __init__(self, dataset_csv_path: pathlib.Path, normalization_values_path: pathlib.Path,
                 return_snr: bool=False, return_sir: bool=False, return_sinr: bool=False,
                 shuffle: bool=False, random_gain_augmentation=False,
                 ):
        super().__init__()

        # load filename list from CSV file
        self.csv_path = dataset_csv_path
        self.files = pd.read_csv(self.csv_path)

        # shuffle the dataframe (once, upon initialization)
        if shuffle:
            print("Shuffling dataset")
            self.files = self.files.sample(frac=1, random_state=base_seed).reset_index(drop=True)

        assert isinstance(self.files, pd.DataFrame), "self.files should be a pandas DataFrame"
        assert 'target' in self.files.columns, "DataFrame must contain 'target' column"
        assert 'noisy' in self.files.columns, "DataFrame must contain 'noisy' column"
        assert len(self) > 0, f"Dataset is empty: {self.csv_path}"

        # Load normalization values
        # need to save it like this: np.savez('normalization_file.npz', mean=mean_array, std=std_array)
        normalization_data = np.load(normalization_values_path)
        self.mean = normalization_data['mean']
        self.std = normalization_data['std']

        assert np.isfinite(self.mean), "Mean value is not finite"
        assert np.isfinite(self.std), "Standard deviation value is not finite"

        self.return_snr = return_snr
        self.return_sir = return_sir
        self.return_sinr = return_sinr

        self.target_column = 'target'
        self.noisy_column = 'noisy'


        self.random_gain_augmentation = random_gain_augmentation
        if self.random_gain_augmentation:
            print("Random gain augmentation enabled!!!")


    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        for each dataset index idx, returns two signals. 
        The first signal is the noisy signal (multi-channel), 
        and the second signal is the target signal (single-channel).

        Args:
            idx (int): Index of the file to load.

        Returns:
            noisy_signal (torch.Tensor): The noisy signal, with shape (channels, samples).
            target_signal (torch.Tensor): The target signal, with shape (1, samples).
        """

        # select file from file list
        filename_target = self.files.iloc[idx][self.target_column]
        filename_noisy = self.files.iloc[idx][self.noisy_column]

        # load files
        target_signal, _ = torchaudio.load(filename_target)
        noisy_signal, _ = torchaudio.load(filename_noisy)

        # Apply normalization scaling 
        # Normalize signals using torch
        target_signal = (target_signal - torch.tensor(self.mean)) / torch.tensor(self.std)
        noisy_signal = (noisy_signal - torch.tensor(self.mean)) / torch.tensor(self.std)

        # (optional) random gain augmentation, in the range of +-10 dB
        if self.random_gain_augmentation:
            
            # generate random gain factor
            factor = 10**(np.random.uniform(-10, 10)/20)
            
            # apply gain factor
            target_signal *= factor
            noisy_signal *= factor

        # Collect the desired SNR/SIR/SINR information if available
        additional_info = self.wrap_additional_info(idx)

        return (noisy_signal, target_signal, *additional_info)
    

    def wrap_additional_info(self, idx):
        """
        Wraps the additional output information.
        """
        additional_info = []
        if self.return_snr and 'snr_db' in self.files.columns:
            additional_info.append(self.files.iloc[idx]['snr_db'])
        if self.return_sir and 'sir_db' in self.files.columns:
            additional_info.append(self.files.iloc[idx]['sir_db'])
        if self.return_sinr and 'sinr_db' in self.files.columns:
            additional_info.append(self.files.iloc[idx]['sinr_db'])

        return additional_info
    

class PreprocessedEnrollmentDataset(PreprocessedDataset):

    def __init__(self, speaker_id_csv_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)

        print(speaker_id_csv_path)

        assert 'clean_inear' in self.files.columns, "DataFrame must contain 'clean_inear' column"
        assert 'talker' in self.files.columns, "DataFrame must contain 'talker' column"

        # read the speaker_id CSV file
        self.speaker_id_df = pd.read_csv(speaker_id_csv_path)

        # Merge the two DataFrames
        self.files = self.files.merge(
            self.speaker_id_df[['audio_file', 'embedding_file']],  # Select only the columns needed
            left_on='target', # column in files
            right_on='audio_file', # column in speaker_id_df 
            how='inner')

        # rename audio_file to enrollment_file
        self.files = self.files.rename(columns={'audio_file': 'enrollment_file'})

        # shuffle the dataframe (once, upon initialization) with a fixed seed
        print("Shuffling dataset")
        self.files = self.files.sample(frac=1, random_state=base_seed).reset_index(drop=True)


    def get_embedding(self, idx: int, filename_target: str) -> torch.Tensor:

        # select all files with the same talker
        talker_id = self.files.iloc[idx]['talker']
        talker_files = self.files[self.files['talker'] == talker_id]

        # select a random file from the same talker different from the current one
        random_idx = np.random.choice(talker_files.index[talker_files['target'] != filename_target])
        enrollment_filename = self.files.iloc[random_idx]['enrollment_file']

        inear_enrollment_filename = enrollment_filename.replace('target_', 'clean_inear_')

        # load enrollment file
        enrollment_signal, _ = torchaudio.load(enrollment_filename)

        inear_enrollment_signal, _ = torchaudio.load(inear_enrollment_filename)

        # Apply normalization scaling 
        enrollment_signal = (enrollment_signal - torch.tensor(self.mean)) / torch.tensor(self.std)
        inear_enrollment_signal = (inear_enrollment_signal - torch.tensor(self.mean)) / torch.tensor(self.std)

        enrollment_signal = torch.cat((enrollment_signal, inear_enrollment_signal), dim=0)

        return enrollment_signal
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # select file from file list
        filename_target = self.files.iloc[idx][self.target_column]
        filename_noisy = self.files.iloc[idx][self.noisy_column]

        # load files
        target_signal, _ = torchaudio.load(filename_target)
        noisy_signal, _ = torchaudio.load(filename_noisy)

        embedding = self.get_embedding(idx, filename_target=filename_target)

        # Apply normalization scaling 
        # Normalize signals using torch
        target_signal = (target_signal - torch.tensor(self.mean)) / torch.tensor(self.std)
        noisy_signal = (noisy_signal - torch.tensor(self.mean)) / torch.tensor(self.std)

        # (optional) random gain augmentation, in the range of +-10 dB
        if self.random_gain_augmentation:
            
            # generate random gain factor
            factor = 10**(np.random.uniform(-10, 10)/20)
            
            # apply gain factor
            target_signal *= factor
            noisy_signal *= factor

        # Collect the desired SNR/SIR/SINR information if available
        additional_info = self.wrap_additional_info(idx)

        return (noisy_signal, target_signal, embedding, *additional_info)