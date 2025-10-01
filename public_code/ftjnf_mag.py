import torch
import torchinfo

from ftjnf import FTJNF


class FTJNF_MAG(FTJNF):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.

    Different from the base class, this model has output type 'IRM' (ideal ratio mask) 
    and uses the STFT magnitude as input.
    """
    def __init__(self, output_type='IRM', **kwargs):
        super().__init__(output_type=output_type, **kwargs)

        lstm_input = self.n_channels

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
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
            
        self.ff = torch.nn.Linear(self.lstm2_out, out_features=self.linear_out_features)


    def get_input_stft(self, x_signal):
        """
        Computes the STFT of the input signal.
        The STFT is computed only for the selected channels,
        The absolute value of the STFT is used as input to the network.

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
        network_input = torch.abs(x_stft)

        return x_stft, network_input


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
                (torch.abs(network_activation), 
                 torch.zeros_like(network_activation)), 
                dim=-1))
        
        # filter and sum in STFT domain (if ref ch is only one channel, should only filter)
        y_stft = torch.sum(
            mask_stft * input_stft[:, [*self.mask_ch], :, :], 
            dim=1, keepdim=False) # output shape (batch, 1, freq, time)
        
        return y_stft, mask_stft 
        

        

if __name__ == '__main__':

    # SE
    model = FTJNF_MAG(n_channels=1, ref_ch=[0], mask_ch=[0])
    
    # AS-SE 
    model = FTJNF_MAG(n_channels=2, ref_ch=[0, 1], mask_ch=[0])

    x = torch.randn(1, 2, 16000)
    
    y, m = model(x, return_mask=True)
    print(y.shape, m.shape)

    torchinfo.summary(model, input_size=x.shape)