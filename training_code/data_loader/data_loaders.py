import pathlib

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from base import BaseDataLoader

from datasets.preprocessed_dataset import PreprocessedDataset,PreprocessedEnrollmentDataset


class PreprocessedDataLoader(BaseDataLoader):
    """
    Preprocessed dataset dataloader using BaseDataLoader
    This dataloader overrides the split_validation method since there is a pre-defined
    split in the dataset.

    Args:
        train_dataset_csv_path: path to the csv file containing the training dataset
        train_normalization_values_path: path to the file containing the normalization values for the training dataset
        validation_dataset_csv_path: path to the csv file containing the validation dataset
        validation_normalization_values_path: path to the file containing the normalization values for the validation dataset
        batch_size: size of the batch
        shuffle: whether to shuffle the dataset
        num_workers: number of workers to use  
    """
    def __init__(self, 
                 train_dataset_csv_path, train_normalization_values_path, 
                 validation_dataset_csv_path, validation_normalization_values_path, 
                 batch_size, shuffle=False, 
                 use_enrollment=False,
                 train_speaker_id_csv_path=None, # only for exceptions
                 validation_speaker_id_csv_path=None, # only for exceptions
                 random_gain_augmentation=False,
                 num_workers=1):
        
        
        train_kwargs = {
            'dataset_csv_path': train_dataset_csv_path,
            'normalization_values_path': train_normalization_values_path,
            'shuffle': shuffle,
            'random_gain_augmentation': random_gain_augmentation
        }

        validation_kwargs = {
            'dataset_csv_path': validation_dataset_csv_path,
            'normalization_values_path': validation_normalization_values_path,
            'shuffle': shuffle,
            'random_gain_augmentation': random_gain_augmentation
        }

        # ENROLLMENT UTTERANCE?
        if use_enrollment:
            dataset_class = PreprocessedEnrollmentDataset

            if train_speaker_id_csv_path is None:
                train_kwargs['speaker_id_csv_path'] = pathlib.Path(train_dataset_csv_path).parent.joinpath("file_paths_train_embeddings.csv")
            else:
                train_kwargs['speaker_id_csv_path'] = train_speaker_id_csv_path

            if validation_speaker_id_csv_path is None:
                validation_kwargs['speaker_id_csv_path'] = pathlib.Path(validation_dataset_csv_path).parent.joinpath("file_paths_validation_embeddings.csv")
            else:
                validation_kwargs['speaker_id_csv_path'] = validation_speaker_id_csv_path

        # REGULAR DATASET
        else:
            dataset_class = PreprocessedDataset

        self.train_dataset = dataset_class(**train_kwargs)
        self.validation_dataset = dataset_class(**validation_kwargs)
        
        super().__init__(self.train_dataset, 
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         validation_split=0.0, # override because preprocessed datasets have a pre-defined split 
                         num_workers=num_workers)
        
        # keywords to hand to the validation loader upon construction when split_validation is called
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }


    def _split_sampler(self, split):
        """
        This function overrides the default split_sampler function in BaseDataLoader.
        For the preprocessed datasets, we have a pre-defined split, so we don't need to split
        and can just return the train and validation samplers for the full datasets.
        """
        train_idx = np.arange(len(self.train_dataset))
        valid_idx = np.arange(len(self.validation_dataset))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler
    

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, dataset=self.validation_dataset,
                              **self.init_kwargs)
        

class PreprocessedTestDataLoader(DataLoader):
    """"
    Slightly trimmed down dataloader without the validation split functionality.
    Does not require two dataset splits, only one.
    By default, this initializes the dataset with return_snr=True for evaluation purposes.
    """
    def __init__(self, dataset_csv_path: pathlib.Path, normalization_values_path: pathlib.Path,
                 speaker_id_csv_path: pathlib.Path = None,
                 batch_size=16, shuffle=False, num_workers=16, collate_fn=default_collate,
                 use_enrollment=False):

        dataset_kwargs = {
            'dataset_csv_path': dataset_csv_path,
            'normalization_values_path': normalization_values_path,
            'shuffle': shuffle,
            'return_snr': True, 
            'return_sir': True, 
            'return_sinr': True
        }

        # ENROLLMENT UTTERANCE?
        if use_enrollment:
            dataset_class = PreprocessedEnrollmentDataset
            if speaker_id_csv_path is None:
                dataset_kwargs['speaker_id_csv_path'] = pathlib.Path(dataset_csv_path).parent.joinpath("file_paths_test_embeddings.csv")
            else:
                dataset_kwargs['speaker_id_csv_path'] = speaker_id_csv_path

        # REGULAR DATASET
        else:
            dataset_class = PreprocessedDataset

        dataset = dataset_class(**dataset_kwargs)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(**self.init_kwargs)