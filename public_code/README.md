# PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables

This repository contains code for the different systems used in the paper 'PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables'.
The architectures are heavily based on FT-JNF and the time-domain Speakerbeam (see links below).
Preprocessing code for the Vibravox and Oldenburg datasets is included as well.

## Links 

- Landing page with audio examples: https://bose.github.io/passe/
- ArXiv preprint: https://doi.org/10.48550/arXiv.2509.20875

- Original FT-JNF repository: https://github.com/sp-uhh/deep-non-linear-filter/
- Orignal time-domain Speakerbeam repository: https://github.com/BUTSpeechFIT/speakerbeam/

## Requirements
`pip install numpy torch torchinfo torchaudio asteroid pandas datasets denoiser soundfile`

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

### Vibravox (Training, in-domain evaluation):
- `download_vibravox.py` downloads the dataset from huggingface. Please download the `speech_clean` and `speechless_noisy` subsets.
- `convert_vibravox_to_wav.py` reads from the huggingface dataset, applies resampling to 16 kHz, and performs channel selection, and saves the dataset again as wave files with a corresponding file list that contains speaker information as well.
This file also creates a channel indexing text file.
- `preprocess_vibravox_TSE.py` can be used to obtain the training and validation sets, which contain all scenarios (speech mixed with noise, speech mixed with interferer, speech mixed with interferer and noise)
- `preprocess_vibravox_TSE_separate_eval_sets.py` can be used to obtain the test sets, which are separated by scenario.

Please note that the in-ear microphone signals for vibravox with interfering talkers are not valid for training and evaluation due to the lack of isolated interfering talker in-ear signals (see paper).

### Oldenburg (Cross-dataset evaluation):
- please download https://doi.org/10.5281/zenodo.10844598 (speech) and https://doi.org/10.5281/zenodo.11196866 (impulse responses).
- You will also need to download the `noise_fullband` part of the DNS5 challenge dataset (https://github.com/microsoft/DNS-Challenge/tree/v5dnschallenge_ICASSP2023)
- `preclean_oldenburg.py` to perform resampling and removal of ambient noise in the recorded own voice speech 
- `simulate_oldenburg_noise.py` to spatialize the DNS challenge noise with the impulse responses
- `simulate_oldenburg_interferers.py` to spatialize the Oldenburg interferers with the impulse responses
- `preprocess_oldenburg_TSE.py` can be used to obtain the training and validation sets, which contain all scenarios (speech mixed with noise, speech mixed with interferer, speech mixed with interferer and noise)
- `preprocess_oldenburg_TSE_separate_eval_sets.py` can be used to obtain the test sets, which are separated by scenario.
