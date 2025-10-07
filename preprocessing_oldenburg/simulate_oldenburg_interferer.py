"""
Simulation of multi-channel interfering speech signals using HRTFs/head-related impulse responses
from the Oldenburg dataset for each talker and direction of arrival (DOA) of the noise source.

Slightly modified version of the original script in https://zenodo.org/records/11196867
"""
import pathlib
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.signal as sig
import soundfile as sf
from tqdm import tqdm

# Also be able to catch warnings as errors, for example invalid divison when normalizing empty audio
# https://stackoverflow.com/questions/5644836/in-python-how-does-one-catch-warnings-as-if-they-were-exceptions
warnings.filterwarnings("error")

# seed all randomness
base_seed = 123
np.random.seed(base_seed)

FS_OUT = 16000
NFFT = 4096

N_DIRECTIONS = 8

TALKER_LIST = [
    'VP_01', 'VP_02', 'VP_04', 'VP_05', 'VP_06', 'VP_10', 'VP_11', 'VP_13', 'VP_14',
    'VP_15', 'VP_18', 'VP_21', 'VP_23', 'VP_24', 'VP_26', 'VP_27', 'VP_28', 'VP_29']

def simulate_point_source(audio_data, ir_matrix, vp_idx):

    # select random impulse response
    rnd_direction_idx = np.random.randint(low=0, high=N_DIRECTIONS)
    random_ir = ir_matrix[vp_idx, rnd_direction_idx, :, :]

    filtered_iem_noise = sig.lfilter(b=random_ir[:, 0], a=1, x=audio_data, axis=0)
    #filtered_oem_noise = sig.lfilter(b=random_ir[:, 1], a=1, x=audio_data, axis=0)
    filtered_oem_noise = audio_data.copy()

    simulated_data = np.stack([filtered_iem_noise, filtered_oem_noise], axis=1)

    return simulated_data


def np_rms(x):
    return np.sqrt(np.mean(np.square(x), axis=0))


def setup_irs(ir_path, select_channels):

    num_talkers = len(TALKER_LIST)
    num_samples = NFFT

    ir_matrix = np.zeros(
        (num_talkers, N_DIRECTIONS, num_samples, 2)
    )

    for t_idx, talker in enumerate(TALKER_LIST):
        for a_idx in range(N_DIRECTIONS):
            file_str = f'data_{a_idx}.npz'
            ir_fname = ir_path.joinpath(talker, file_str)
            meas_data = np.load(ir_fname)
            fs = meas_data['samplerate']

            ir_mic = meas_data['ir_mic']
            ir_mic = sig.resample_poly(up=FS_OUT, down=fs, x=ir_mic, axis=0)

            ir_mic_cut = ir_mic[:NFFT, select_channels]

            ir_matrix[t_idx, a_idx, :, :] = ir_mic_cut

    return ir_matrix


def main(path_speech, output_folder, ir_path, reference_channel, sensor_channel, 
         time_secs=10, num_processes=32):

    # new ch0 is reference/outer mic, ch1 is in-ear/other sensor mic
    select_channels = [reference_channel, sensor_channel]
    select_new_inear_ch = 1

    ir_matrix = setup_irs(ir_path, select_channels)

    for talker_str in TALKER_LIST:
        output_folder.joinpath(talker_str).mkdir(exist_ok=True)

    file_list = [fname for fname in path_speech.rglob('*.wav')]
    
    # filter out files where the subfolder is not in TALKER_LIST (i think that's also done before)
    file_list = [fname for fname in file_list if fname.parent.stem in TALKER_LIST]

    # shuffle file list and select only a fraction of the files
    np.random.shuffle(file_list)

    args = [(example, base_seed, i, ir_matrix, select_new_inear_ch, 
             output_folder, time_secs, reference_channel
             ) for i, example in enumerate(file_list)] 
    
    with Pool(processes=num_processes, initializer=np.random.seed) as pool:
        _ = list(tqdm(pool.imap(process_example, args), total=len(args)))
    
    print('finished!')


def process_example(args):

    (fname, base_seed, i, ir_matrix, select_new_inear_ch, output_folder, time_secs, ref_ch) = args
    
    # seed all randomness for this process
    np.random.seed(base_seed + i)

    audio_data, fs = sf.read(fname, always_2d=True)
    audio_data = sig.resample_poly(up=FS_OUT, down=fs, x=audio_data, axis=0)

    audio_data = audio_data[:, ref_ch]

    # get the "listener" string to exclude it
    listener_str = fname.stem.split('_')[0]

    # exclude it
    talker_excluded_list = [talker for talker in TALKER_LIST if talker != listener_str]

    # randomly choose talker index
    vp_idx_temp = np.random.randint(low=0, high=len(talker_excluded_list))
    vp_idx = TALKER_LIST.index(talker_excluded_list[vp_idx_temp])

    # simulate point source
    simulated_data = simulate_point_source(audio_data, ir_matrix, vp_idx)

    # add small amount of white noise to IEM
    randscale = np.random.rand(1) * 0.001 * np_rms(simulated_data[:, select_new_inear_ch])
    rand_whitenoise = np.random.randn(*simulated_data[:, select_new_inear_ch].shape)
    rand_whitenoise = randscale * rand_whitenoise / np_rms(rand_whitenoise)

    simulated_data[:, select_new_inear_ch] += rand_whitenoise

    # scale audio to maximum amplitude of 1
    try:
        simulated_data /= np.amax(np.abs(simulated_data))
    except RuntimeWarning:
        print(f'skipping file: {fname}, something weird happened (division by zero?)')
        pass

    out_fname = output_folder.joinpath(TALKER_LIST[vp_idx], fname.name)

    sf.write(file=out_fname, data=simulated_data, samplerate=FS_OUT)


if __name__ == '__main__':

    path_speech = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/speech_precleaned')

    output_folder = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/simulated_interferer_OL')
    print(f"warning: writing files without spatialized outer mic to {output_folder}")
    
    ir_path = pathlib.Path('/home/ubuntu/data/hearpiece_ownvoice/hrtf_measurements')

    output_folder.mkdir(exist_ok=True)

    # 0: Lavalier-microphone clipped to the shirt neck, shirt collar etc. of the talker
    # 1: Reference microphone about 50 cm in front of the talker
    # 2: Left in-ear microphone Hearpiece
    # 3: Left concha microphone Hearpiece
    # 4: Right in-ear microphone Hearpiece
    # 5: Right concha microphone Hearpiece

    setup_choices = {
        'outer_inear': {
            'reference_channel': 3, 
            'sensor_channel': 2
            },
        'outer_inear_right_side': {
            'reference_channel': 5, 
            'sensor_channel': 4
            },
        'closetalk_inear': {
            'reference_channel': 1, 
            'sensor_channel': 2
            },
        'closetalk_inear_right_side': {
            'reference_channel': 1, 
            'sensor_channel': 4
            } 
         
        # add other combinations if desired
        }
    
    time_secs = 3 

    num_processes = 32

    #raise ValueError("YOU NEED TO REMEMBER TO DELETE THE OLD FILES FIRST!!!")
    
    # SELECT HERE
    for setup_name in ['outer_inear', 'outer_inear_right_side']:
    
        reference_channel = setup_choices[setup_name]['reference_channel']
        sensor_channel = setup_choices[setup_name]['sensor_channel']

        output_folder_setup = output_folder.joinpath(setup_name)
        output_folder_setup.mkdir(exist_ok=True)

        main(path_speech=path_speech,
            output_folder=output_folder_setup,
            ir_path=ir_path,
            reference_channel=reference_channel,
            sensor_channel=sensor_channel,
            time_secs=time_secs,
            num_processes=num_processes)