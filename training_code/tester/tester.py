
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


class Tester():

    def __init__(self, model, metric_functions, device, dataloader,
                 test_output_path, results_filename, example_path):
        
        self.test_output_path = test_output_path
        self.results_filename = results_filename
        self.example_path = example_path
        
        self.model = model
        self.metric_functions = metric_functions

        self.device = device
        self.dataloader = dataloader


    def test_full_dataset(self):
        
        # preparations
        self.model.eval()

        results = []

        with torch.no_grad():

            # iterate over full dataset once
            for batch_idx, batch_data in enumerate(tqdm(self.dataloader, desc="testing")):
                data, target = batch_data[:2]
                additional_info = batch_data[2:] if len(batch_data) > 2 else []

                if len(additional_info) == 1:
                    snr_db = additional_info[0]
                    sir_db = None
                    sinr_db = None
                elif len(additional_info) == 2:
                    snr_db, sir_db = additional_info
                    sinr_db = None
                elif len(additional_info) == 3:
                    snr_db, sir_db, sinr_db = additional_info
                else:
                    snr_db = None
                    sir_db = None
                    sinr_db = None
                
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                batch_results = {'snr_db': snr_db.tolist()}
                if sir_db is not None:
                    batch_results['sir_db'] = sir_db.tolist()
                if sinr_db is not None:
                    batch_results['sinr_db'] = sinr_db.tolist()

                # Compute results for this batch
                for met_name, met in self.metric_functions:
                    result = met(output, target)

                    # if result is a single value per example, save it directly
                    if met_name in ['si_sdr_per_example', 'estoi', 'pesq_wb']:
                        metric_key = met_name

                        if metric_key not in batch_results:
                            batch_results[metric_key] = []
                        batch_results[metric_key].extend(r.item() if hasattr(r, 'item') else r for r in result)
                    else:

                        # create metric sub-keys for each value (per example) in the result
                        for sub_idx in range(result.shape[1]):
                            metric_key = f'{met_name}_{sub_idx}'

                            if metric_key not in batch_results:
                                batch_results[metric_key] = []

                            batch_results[metric_key].extend(result[:, sub_idx].tolist())

                # Convert batch_results to individual example results
                for i in range(len(data)):

                    example_result = {'snr_db': batch_results['snr_db'][i]}
                    if sir_db is not None:
                        example_result['sir_db'] = batch_results['sir_db'][i]
                    if sinr_db is not None:
                        example_result['sinr_db'] = batch_results['sinr_db'][i]

                    for key in batch_results:
                        if key not in ['snr_db', 'sir_db', 'sinr_db']:
                            example_result[key] = batch_results[key][i]
                    results.append(example_result)

                if batch_idx == 0:
                    additional_info = []
                    if snr_db is not None:
                        additional_info.append(snr_db)
                    if sir_db is not None:
                        additional_info.append(sir_db)
                        
                    # save the entire batch as audio examples
                    self.save_example_batch(data, output, target, 
                                            additional_info=additional_info)


        # create dataframe
        df = pd.DataFrame(results)
        df.to_csv(self.results_filename, index=True)




    def save_example_batch(self, data, output, target, additional_info):
        """
        Save all examples from a batch (noisy, output, target) from the batch to audio files.

        :param data: Tensor, input data (noisy).
        :param output: Tensor, model output.
        :param target: Tensor, target data.
        :param additional_info: input SNR in dB, input SIR in dB, input SINR in dB (all optional).
        """
        for b in range(data.shape[0]):
        # Save the first example
            noisy_ex = data[b, ...].cpu()
            output_ex = output[b, ...].cpu()
            target_ex = target[b, ...].cpu()

            snr_db = additional_info[0].clone().detach().float()
            if len(additional_info) > 1:
                sir_db = additional_info[1].clone().detach().float()
            else:
                sir_db = None

            # compute maximum amplitude of noisy signal
            norm_val = torch.amax(torch.abs(noisy_ex))

            snr_sir_string = f'snr_{torch.round(snr_db[b], decimals=2)}dB'
            if sir_db is not None:
                snr_sir_string += f'_sir_{torch.round(sir_db[b], decimals=2)}dB'

            fname_noisy = self.example_path.joinpath(f'noisy_{b}_{snr_sir_string}.wav')
            fname_output = self.example_path.joinpath(f'output_{b}_{snr_sir_string}.wav')
            fname_target = self.example_path.joinpath(f'target_{b}_{snr_sir_string}.wav')

            torchaudio.save(fname_noisy, noisy_ex / norm_val, 16000)
            torchaudio.save(fname_output, output_ex / norm_val, 16000)
            torchaudio.save(fname_target, target_ex / norm_val, 16000)