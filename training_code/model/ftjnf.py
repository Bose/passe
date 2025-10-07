# Description: PyTorch implementation of the FTJNF model
# taken from here: https://github.com/sp-uhh/deep-non-linear-filter/blob/main/src/models/models.py
# changes: add STFT into the model, change input and output shapes, add inverse STFT
# also default for bidrectional is False now

from typing import Literal

import torch
import torchaudio
import torchinfo


class FTJNF(torch.nn.Module):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.
    """
    def __init__(self,
                 n_channels: int,
                 ref_ch: int = [0],
                 n_lstm_hidden1: int = 512,
                 n_lstm_hidden2: int = 128,
                 bidirectional: bool = False,
                 output_type: Literal['IRM', 'CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh',
                 mask_ch=None):
        """
        Initialize model.

        :param n_channels: number of channel in the input signal
        :param ref_ch: index of the reference channel for which an STFT mask is estimated
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param bidirectional: set to True for a bidirectional LSTM
        :param output_type: output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.bidirectional = bidirectional
        self.output_type = output_type
        self.output_activation = output_activation

        # indices of input channels and output masks
        if mask_ch is not None:
            self.mask_ch = mask_ch
        else:
            self.mask_ch = ref_ch
        self.ref_ch = ref_ch

        lstm_input = 2*self.n_channels

        self.lstm1 = torch.nn.LSTM(input_size=lstm_input, 
                                   hidden_size=self.n_lstm_hidden1, 
                                   bidirectional=self.bidirectional, 
                                   batch_first=False)
        
        self.lstm1_out = 2*self.n_lstm_hidden1 if self.bidirectional else self.n_lstm_hidden1
        lstm2_input = self.lstm1_out

        self.lstm2 = torch.nn.LSTM(input_size=lstm2_input, 
                                   hidden_size=self.n_lstm_hidden2, 
                                   bidirectional=self.bidirectional, 
                                   batch_first=False)
        
        self.lstm2_out = 2*self.n_lstm_hidden2 if self.bidirectional else self.n_lstm_hidden2

        if self.output_type == 'IRM':
            self.linear_out_features = 1 * len(self.mask_ch)
        elif self.output_type == 'CRM':
            self.linear_out_features = 2 * len(self.mask_ch)
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
            
        self.ff = torch.nn.Linear(self.lstm2_out, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = torch.nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = torch.nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = torch.nn.Identity()

        self.nfft = 512

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft, hop_length=self.nfft//2, win_length=self.nfft, power=None)
        
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.nfft, hop_length=self.nfft//2, win_length=self.nfft)


    def get_input_stft(self, x_signal):
        """
        Computes the STFT of the input signal.
        The STFT is computed only for the selected channels,
        and the real and imaginary parts are stacked in the channel dimension.

        Args:
            x_signal: input signal with shape [BATCH, CHANNEL, TIME].

        Returns:
            the STFT of the input signal with shape [BATCH, CHANNEL*2, FREQ, TIME].
        """
        # select channels
        x_signal = x_signal[:, self.ref_ch, :] # shape (batch, ch, time)

        # Compute STFT for noisy signals
        x_stft = self.stft(x_signal)

        # split the complex spectrogram into its real and imaginary parts, 
        # and stack them in the channel dimension
        network_input = torch.cat([x_stft.real, x_stft.imag], dim=1)

        return x_stft, network_input


    def forward(self, input_signal: torch.Tensor, 
                conditioning_vector: torch.tensor=None,
                return_mask: bool=False) -> torch.Tensor:
        """
        Implements the forward pass of the model.
        Modified, now input is expected to be a time-domain signal, and the output is the filtered signal in the time-domain.

        Args:
            x_signal: input signal with shape [BATCH, CHANNEL, TIME].

        Returns:
            the output signal with shape [BATCH, 1, TIME].
        """
        input_length = input_signal.shape[-1]
        
        input_stft, network_input = self.get_input_stft(input_signal)

        # :param x: input with shape [BATCH, CHANNEL, FREQ, TIME], 
        # where channel is 2x the actual microphone channels (real and imaginary parts)
        n_batch, n_channel, n_freq, n_times = network_input.shape

        # wide_band
        x = network_input.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        x, _ = self.lstm1(x)

        # wide_band -> narrow_band
        x =  x.reshape(n_freq, n_batch, n_times, self.lstm1_out).permute(2,1,0,3).reshape(n_times, n_batch*n_freq, self.lstm1_out)

        x, _ = self.lstm2(x)

        x = self.ff(x)

        # narrow_band -> input shape
        x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        
        network_activation = self.mask_activation(x) # shape (batch, 1 or 2, freq, time)
        
        output_stft, mask_stft = self.compute_output_stft(network_activation, input_stft) # shape (batch, 1, freq, time)

        output_signal = self.compute_time_domain_signal(output_stft, input_length) # shape (batch, 1, time)

        if return_mask:
            return output_signal, mask_stft
        else:
            return output_signal 
        

    def compute_output_stft(self, network_activation: torch.tensor, input_stft: torch.tensor) -> torch.tensor:
        """
        Computes the output STFT from the network activation.

        Args:
            network_activation: output of the network with shape [BATCH, CHANNEL*2, FREQ, TIME].
            input_stft: complex-valued STFT of the input signal with shape [BATCH, CHANNEL, FREQ, TIME].

        Returns:
            the output STFT with shape [BATCH, 1, FREQ, TIME].
        """
        # from real and imaginary parts, re-assemble the complex spectrogram
        mask_stft = torch.view_as_complex(
            torch.stack(
                (network_activation[:, 0::2, :, :], 
                 network_activation[:, 1::2, :, :]), 
                dim=-1))
        
        # filter and sum in STFT domain (if ref ch is only one channel, should only filter)
        y_stft = torch.sum(
            mask_stft * input_stft[:, [*self.mask_ch], :, :], 
            dim=1, keepdim=False) # output shape (batch, 1, freq, time)
        
        return y_stft, mask_stft 
        

    def compute_time_domain_signal(self, y_stft: torch.tensor, input_length: int) -> torch.tensor:
        """
        Computes the time-domain signal from the STFT of the signal.

        Args:
            y_stft: STFT of the signal with shape [BATCH, CHANNEL, FREQ, TIME].

        Returns:
            the time-domain signal with shape [BATCH, 1, TIME].
        """
        # istft back to time domain
        y = self.istft(y_stft, input_length) # shape (batch, time)

        # add new channel dimension 
        return y[:, None, :] # shape (batch, 1, time)
    

    