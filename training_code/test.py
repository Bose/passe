import argparse
import os
import pathlib
from pathlib import PosixPath

import torch

import data_loader.data_loaders as module_data
import model as module_arch
import model.metric as module_metric

#from tester import Tester
import tester as module_tester
from parse_config import ConfigParser
from utils import prepare_device, read_json


def init_test_obj(config: dict, object_type: str, object_module, **kwargs):
    """
    Init an object needed for testing based on configuration dictionary.

    Args:
        config (dict): a dictionary containing at least the key object_type
                    with subfields "type" and "args" for construction
        object_type (str): the type of object to construct
        object_module (): the class to construct an object of
    """
    object_args = dict(config[object_type]['args'])

    return getattr(object_module, config[object_type]['type'])(**object_args, **kwargs)


def main(args):

    # build model architecture
    # from the model_file and correspondig configuration file
    if pathlib.Path(args.model_file).suffix == '.pth':
        model_config_fname = pathlib.Path(args.model_file).parent.joinpath('config.json')
        model_config = read_json(model_config_fname)
    elif pathlib.Path(args.model_file).suffix == '.json':
        model_config = read_json(args.model_file)
    else:
        raise ValueError("Model file must be either a .pth (trained model) or .json (dummies) file.")
    model = init_test_obj(model_config, 'arch', module_arch)


    # Load the test configuration file
    test_config = read_json(args.test_config)

    # setup dataset instance
    #dataloader = init_test_obj(test_config, 'data_loader', module_data, 
    #    normalization_values_path=model_config['data_loader']['args']['train_normalization_values_path'])
    dataloader = init_test_obj(test_config, 'data_loader', module_data)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # prepare for (multi-device) GPU testing
    device, device_ids = prepare_device(test_config['n_gpu'])
    model = model.to(device)

    # if the model is a dummy, do nothing, otherwise load weights
    if model_config['arch']['type'] == 'DummyModel':
        print("Model is a dummy model, no weights to load.") 
    else:
        # load stored weights from trained model
        # (since the template saves some other stuff, we need to specify weights_only=True
        # and add the necessary classes to the safe globals so that torch.load can load the model)
        torch.serialization.add_safe_globals([ConfigParser])
        torch.serialization.add_safe_globals([PosixPath])
        checkpoint = torch.load(args.model_file, weights_only=True)
        state_dict = checkpoint['state_dict']
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.load_state_dict(state_dict)
                
        else:
            raise NotImplementedError("Single GPU testing not implemented yet.")
            
    # get function handles of loss and metrics (and the string names)
    metric_fns = [(met, getattr(module_metric, met)) for met in test_config['metrics']]

    if model_config['arch']['type'] == 'DummyModel':
        output_path = pathlib.Path('saved', 'models', model_config["name"]).joinpath('test_results')
    else:
        output_path = pathlib.Path(args.model_file).parent.joinpath('test_results')
    output_path.mkdir(exist_ok=True)
    
    testname = pathlib.Path(args.test_config).stem
    test_output_path = output_path.joinpath(testname)
    test_output_path.mkdir(exist_ok=True)

    example_path = test_output_path.joinpath('audio_examples')
    example_path.mkdir(exist_ok=True)

    results_filename = test_output_path.joinpath('results_per_file.csv')

    tester = init_test_obj(test_config, 'tester', module_tester,
                           model=model, metric_functions=metric_fns, device=device, 
                           dataloader=dataloader, test_output_path=test_output_path,
                           results_filename=results_filename, example_path=example_path)

    tester.test_full_dataset()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('-m', '--model_file', default=None, type=str,
                        help='model file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--test_config', default=None, type=str,
                        help='test config file path (default: None)')

    args = parser.parse_args()

    main(args)

    # example call: python test.py -m saved/models/Vibravox_PASSE_IM_D/0123_456789/model_best.pth -t test_configs/Vibravox_I_test.json 
