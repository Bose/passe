# PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables
This repository contains code for the paper 'PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables'.
The architectures are heavily based on FT-JNF and the time-domain Speakerbeam (see links below).
Preprocessing and training code for the Vibravox and Oldenburg datasets is included as well.


## Links related to this paper
- Landing page with audio examples: https://bose.github.io/passe/
- ArXiv preprint: https://doi.org/10.48550/arXiv.2509.20875

## Links to code we used
- Original FT-JNF repository: https://github.com/sp-uhh/deep-non-linear-filter/
- Orignal time-domain Speakerbeam repository: https://github.com/BUTSpeechFIT/speakerbeam/

- The training code in this repository is based on this template: https://github.com/victoresque/pytorch-template

Many thanks to the original authors!


## Requirements
Python packages and versions used for this project are listed in `requirements.txt`.

The code and training configuration files were used on an AWS instance with `--instance-type g6.24xlarge` (see https://aws.amazon.com/ec2/instance-types/) so it might be necessary to adjust number of workers, batch sizes and number of GPUs to your hardware to be able to retrain the models.


## Example model usage
This is how to initialize the DNN models used in the paper:

```python
    # assumed channel order here is 0: 'outer microphone'
    #                               1: 'in-ear microphone'

    e = torch.randn(1, 2, 3*16000)  # enrollment signal with 2 channels, each 3 seconds long
    x = torch.randn(1, 2, 3*16000)  # input signal with 2 channels, each 3 seconds long

    # SE
    model = FTJNF_MAG(n_channels=1, ref_ch=[0], mask_ch=[0])

    # PSE with OM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=1, ref_ch=[0], cond_vector_size=192, mask_ch=[0], cond_vector_ch=0)
    
    # PSE with IM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=1, ref_ch=[0], cond_vector_size=192, mask_ch=[0], cond_vector_ch=1)
    
    # AS-SE 
    model = FTJNF_MAG(n_channels=2, ref_ch=[0, 1], mask_ch=[0])
    
    # PASSE with OM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=2, ref_ch=[0,1], cond_vector_size=192, mask_ch=[0], cond_vector_ch=0)
    
    # PASSE with IM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=2, ref_ch=[0,1], cond_vector_size=192, mask_ch=[0], cond_vector_ch=1)
    

    y = model(x, e)
    print(y.shape)

    torchinfo.summary(model, input_size=(x.shape, e.shape))
```


## Dataset pre-processing workflows
Preprocessing files are located in the `preprocessing_vibravox` and `preprocessing_oldenburg` folders.
Before you run any of the files, please review them and make sure to change paths based on your system directory structure.


### Vibravox (Training, in-domain evaluation)
Files located in the folder `preprocessing_vibravox/`.
- `download_vibravox.py` downloads the dataset from huggingface. Please download the `speech_clean` and `speechless_noisy` subsets.
- `convert_vibravox_to_wav.py` reads from the huggingface dataset, applies resampling to 16 kHz, and performs channel selection, and saves the dataset again as wave files with a corresponding file list that contains speaker information as well.
This file also creates a channel indexing text file.
- `preprocess_vibravox_TSE.py` can be used to obtain the training and validation sets, which contain all scenarios (speech mixed with noise, speech mixed with interferer, speech mixed with interferer and noise). Please note that the in-ear microphone signals for vibravox with interfering talkers are not valid for training and evaluation due to the lack of isolated interfering talker in-ear signals (see paper).
- `preprocess_vibravox_TSE_scaled_outer_interferer.py` can be used to obtain the training and validation sets where the corresponding interfering talker in-ear signals are approximated by a scaling factor (Configuration (D) in the paper).
- `preprocess_vibravox_TSE_separate_eval_sets.py` can be used to obtain the test sets, which are separated by scenario.
- `prepare_vibravox_enrollments.py` to create a table of enrollment file names. The actual selection takes place in the training dataset code. You might need to uncomment the correct splits and change paths here too.


### Oldenburg (Cross-dataset evaluation)
Files located in the folder `preprocessing_oldenburg/`.
- please download https://doi.org/10.5281/zenodo.10844598 (speech) and https://doi.org/10.5281/zenodo.11196866 (impulse responses).
- You will also need to download the `noise_fullband` part of the DNS5 challenge dataset (using the script provided in https://github.com/microsoft/DNS-Challenge/tree/v5dnschallenge_ICASSP2023)
- `preclean_oldenburg.py` to perform resampling and removal of ambient noise in the recorded own voice speech 
- `simulate_oldenburg_noise.py` to spatialize the DNS challenge noise with the impulse responses
- `simulate_oldenburg_interferers.py` to spatialize the Oldenburg interferers with the impulse responses
- `preprocess_oldenburg_TSE.py` can be used to obtain the training and validation sets, which contain all scenarios (speech mixed with noise, speech mixed with interferer, speech mixed with interferer and noise)
- `preprocess_oldenburg_TSE_separate_eval_sets.py` can be used to obtain the test sets, which are separated by scenario.
- `prepare_oldenburg_enrollments.py` to create a table of enrollment file names. The actual selection takes place in the training dataset code. You might need to uncomment the correct splits and change paths here too.


## Training
After generating the data required for training, you can train a model using one of the configuration files provided in the folder `training_code/configs/`. You probably want to change the file paths for the pre-processed data here, too.

Example call: `train.py -c configs/Vibravox_SE.json -d 0,1` for training the basic single-channel unconditioned system on the Vibravox dataset, using GPUs 0 and 1.


## Bibtex
If you found this code helpful or want to reference it, please cite as:
```
@article{passe2025,
  title = {{PAS-SE}: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables},
  journal = {arXiv:2509.20875},
  author = {Ohlenbusch, Mattes and Kegler, Mikolaj and Stamenovic, Marko},
  year = {2025},
  month = sep,
  doi = {10.48550/arXiv.2509.20875}
}
```