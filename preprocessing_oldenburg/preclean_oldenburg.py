"""
Precleaning script for the Oldenburg dataset.
This script uses a pretrained denoiser model (DEMUCS) to clean the audio files.
"""
import pathlib

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from tqdm import tqdm


def main():

    input_dir = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/speech/')
    output_dir = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/speech_precleaned/')
    output_dir.mkdir(exist_ok=True)

    clean_channels = [3, 5]

    model = pretrained.master64().cuda() # expects 16kHz audio


    for wav_example_path in tqdm(input_dir.glob('**/*.wav')):
        mc_wav, sr = torchaudio.load(wav_example_path)
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        mc_wav = resample(mc_wav)
        sr = 16000

        for ch in clean_channels:

            norm_val = torch.max(torch.abs(mc_wav[[ch], :]))
            wav = mc_wav[[ch],:] / norm_val

            wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
            with torch.no_grad():
                denoised = model(wav[None])[0]
            denoised = denoised.data.cpu()

            mc_wav[[ch], :] = denoised * norm_val

        # Create the necessary subdirectories in the output directory
        relative_path = wav_example_path.relative_to(input_dir)
        output_file_path = output_dir.joinpath(relative_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(output_file_path, mc_wav, model.sample_rate)

if __name__ == '__main__':
    main()