import torch
import torchaudio
import torchinfo
from asteroid.masknn.convolutional import TDConvNet
from asteroid_filterbanks import make_enc_dec

from model.ftjnf_mag import FTJNF_MAG


class MultConditioning(torch.nn.Module):
    def __init__(self, input_output_size, cond_vector_size=128):
        super().__init__()

        # conditioning vectors dimensioned so that they can be multiplied 
        # with the input and maintain its dimensionality
        self.fc_gamma = torch.nn.Linear(cond_vector_size, input_output_size)

    def forward(self, x, conditioning_vector):

        # parameters
        gamma = self.fc_gamma(conditioning_vector)

        # apply the multiplier to the input x
        return x * gamma[None, :, None, :] 
    

class Lambda(torch.nn.Module):
    """
    https://stackoverflow.com/a/64064088
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x): return self.func(x, **self.kwargs)



class SpeakerBeamEncoder(torch.nn.Module):
    """
    SpeakerBeamEncoder is a simple feedforward neural network that takes a conditioning vector as input and outputs
    a conditioning vector of the same size.
    """

    def __init__(self, output_size: int = 128, sample_rate: int = 8000, fb_name: str = "free"):
        super().__init__()

        if fb_name == "stft_mag":
            n_banks_enc = 257
        else:
            n_banks_enc = 512

        self.register_buffer(
            "stft_window",
            torch.hann_window(512, periodic=True)  # Default device is CPU
        )

        self.sample_rate = sample_rate
        if self.sample_rate < 16000:
            print(f'WARNING: sample_rate {self.sample_rate} is lower than 16000, manual band-limitation applied to speaker encoder')
            self.do_bandlim = True
            self.max_freq_idx = int(257 * self.sample_rate // (16000 // 2))
            print(self.max_freq_idx)
        else:
            self.do_bandlim = False

        if fb_name == "stft_mag":


            stft_obj = torchaudio.transforms.Spectrogram(
                n_fft=512, hop_length=256, win_length=512, power=None)
            
            encoder_aux = torch.nn.Sequential(
                Lambda(torch.squeeze, dim=1),
                stft_obj,
                Lambda(torch.abs),
                )
        else:
            # Encoder for auxiliary network
            encoder_aux, _ = make_enc_dec(
                fb_name=fb_name,
                kernel_size=16,
                n_filters=n_banks_enc,
                stride=8,
                sample_rate=sample_rate 
            )
        
        # Auxiliary network
        self.auxiliary = torch.nn.Sequential(
            Lambda(torch.unsqueeze, dim=1),
            encoder_aux,
            TDConvNet(
                in_chan=n_banks_enc,
                n_src=1,
                out_chan=output_size,
                n_blocks=8,
                n_repeats=1,
                bn_chan=128,
                hid_chan=512,
                skip_chan=128,
                conv_kernel_size=3,
                norm_type="gLN",
                mask_act='linear',
                causal=False
            ),
            Lambda(torch.mean, dim=-1),
            Lambda(torch.squeeze, dim=1)
        )

    def forward(self, enrollment_signal: torch.Tensor) -> torch.Tensor:
        # input: (batch, ch, time)
        # output: (batch, 1, embedding_size)

        if self.do_bandlim:
            enrollment_stft = torch.stft(
                enrollment_signal,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=self.stft_window,
                return_complex=True
            )
            # keep only the frequencies of the spectrum up to half the sample rate
            enrollment_stft[:, :, self.max_freq_idx:] = 0.0

            enrollment_signal = torch.istft(
                enrollment_stft,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=self.stft_window,
                length=enrollment_signal.shape[-1]
            )

        return self.auxiliary(enrollment_signal)


class FTJNF_MAG_1mult(FTJNF_MAG):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.

    The network is conditioned on a conditioning vector, which is used to modulate the output of the LSTM layers.
    """
    def __init__(self,cond_vector_size: int=128, cond_vector_ch: int=0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.cond_vector_ch = cond_vector_ch
        self.cond_vector_size = cond_vector_size
        if isinstance(self.cond_vector_ch, list):
            # multiply size of conditioning vector by number of channels
            self.cond_vector_size *= len(self.cond_vector_ch)

        self.film1 = MultConditioning(self.lstm1_out, cond_vector_size=self.cond_vector_size)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, input_signal: torch.Tensor, conditioning_vector: torch.tensor) -> torch.Tensor:
        """
        Implements the forward pass of the model.
        Modified, now input is expected to be a time-domain signal, and the output is the filtered signal in the time-domain.

        Args:
            x_signal: input signal with shape [BATCH, CHANNEL, TIME].

        Returns:
            the output signal with shape [BATCH, 1, TIME].
        """
        # select the conditioning vector for the desired channel(s)
        if isinstance(self.cond_vector_ch, list):
            # multiple channels - squash them together 
            conditioning_vector = torch.cat([conditioning_vector[:, ch, :] for ch in self.cond_vector_ch], dim=-1)
        else:
            conditioning_vector = conditioning_vector[:, self.cond_vector_ch, :]

        conditioning_vector = self.dropout(conditioning_vector)

        input_length = input_signal.shape[-1]
        
        input_stft, network_input = self.get_input_stft(input_signal)

        # :param x: input with shape [BATCH, CHANNEL, FREQ, TIME], 
        # where channel is 2x the actual microphone channels (real and imaginary parts)
        n_batch, n_channel, n_freq, n_times = network_input.shape

        # wide_band
        x = network_input.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        x, _ = self.lstm1(x)

        # wide_band -> narrow_band
        x = x.reshape(n_freq, n_batch, n_times, self.lstm1_out)
        x = self.film1(x, conditioning_vector)
        x = x.permute(2,1,0,3)
        x = x.reshape(n_times, n_batch*n_freq, self.lstm1_out)

        x, _ = self.lstm2(x)

        x = self.ff(x)

        # narrow_band -> input shape
        x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        
        network_activation = self.mask_activation(x) # shape (batch, 1 or 2, freq, time)
        
        output_stft, _ = self.compute_output_stft(network_activation, input_stft) # shape (batch, 1, freq, time)

        output_signal = self.compute_time_domain_signal(output_stft, input_length) # shape (batch, 1, time)

        return output_signal 


class FTJNF_MAG_1mult_learned_enc(FTJNF_MAG_1mult):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.

    The network is conditioned on a conditioning vector, which is used to modulate the output of the LSTM layers.
    """
    def __init__(self,cond_vector_size: int=128, cond_vector_ch: int=0, dropout=0.0, encoder_sample_rate=16000, 
                 encoder_fb_name: str='free', **kwargs):
        super().__init__(**kwargs)

        self.cond_vector_ch = cond_vector_ch
        self.cond_vector_size = cond_vector_size

        self.embedding_encoder = SpeakerBeamEncoder(output_size=self.cond_vector_size,
                                                    sample_rate=encoder_sample_rate,
                                                    fb_name=encoder_fb_name)
        
        if isinstance(self.cond_vector_ch, list): # 2 channels -> need escond encoder
            assert len(self.cond_vector_ch) == 2, "Only 2 channels are supported"
            self.embedding_encoder2 = SpeakerBeamEncoder(output_size=self.cond_vector_size,
                                                        sample_rate=encoder_sample_rate,
                                                        fb_name=encoder_fb_name)
            
        self.cond_input_size = self.cond_vector_size
        if isinstance(self.cond_vector_ch, list):
            # multiply size of conditioning vector by number of channels
            self.cond_input_size *= len(self.cond_vector_ch)
            
        self.film1 = MultConditioning(self.lstm1_out, cond_vector_size=self.cond_input_size)


    def forward(self, input_signal: torch.Tensor, enrollment_signal: torch.tensor) -> torch.Tensor:
        """
        Implements the forward pass of the model.
        Modified, now input is expected to be a time-domain signal, and the output is the filtered signal in the time-domain.

        Args:
            x_signal: input signal with shape [BATCH, CHANNEL, TIME].

        Returns:
            the output signal with shape [BATCH, 1, TIME].
        """
        # get the conditioning vector
        if isinstance(self.cond_vector_ch, list):
            # get the conditioning vector for each channel
            conditioning_vector = self.embedding_encoder(enrollment_signal[:, self.cond_vector_ch[0], :])
            conditioning_vector2 = self.embedding_encoder2(enrollment_signal[:, self.cond_vector_ch[1], :])
            conditioning_vector = torch.cat((conditioning_vector, conditioning_vector2), dim=1)
        else:
            # get the conditioning vector for the first channel
            conditioning_vector = self.embedding_encoder(enrollment_signal[:, self.cond_vector_ch, :])

        input_length = input_signal.shape[-1]
        
        input_stft, network_input = self.get_input_stft(input_signal)

        # :param x: input with shape [BATCH, CHANNEL, FREQ, TIME], 
        # where channel is 2x the actual microphone channels (real and imaginary parts)
        n_batch, n_channel, n_freq, n_times = network_input.shape

        # wide_band
        x = network_input.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        x, _ = self.lstm1(x)

        # wide_band -> narrow_band
        x = x.reshape(n_freq, n_batch, n_times, self.lstm1_out)
        x = self.film1(x, conditioning_vector)
        x = x.permute(2,1,0,3)
        x = x.reshape(n_times, n_batch*n_freq, self.lstm1_out)

        x, _ = self.lstm2(x)

        x = self.ff(x)

        # narrow_band -> input shape
        x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        
        network_activation = self.mask_activation(x) # shape (batch, 1 or 2, freq, time)
        
        output_stft, _ = self.compute_output_stft(network_activation, input_stft) # shape (batch, 1, freq, time)

        output_signal = self.compute_time_domain_signal(output_stft, input_length) # shape (batch, 1, time)

        return output_signal 
    

if __name__ == '__main__':

    # PSE with IM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=1, ref_ch=[0], cond_vector_size=192, mask_ch=[0], cond_vector_ch=0, encoder_sample_rate=16000)
    
    # PSE with IM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=1, ref_ch=[0], cond_vector_size=192, mask_ch=[0], cond_vector_ch=1, encoder_sample_rate=16000)
    
    # PASSE with OM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=2, ref_ch=[0,1], cond_vector_size=192, mask_ch=[0], cond_vector_ch=0, encoder_sample_rate=16000)
    
    # PASSE with IM conditioning
    model = FTJNF_MAG_1mult_learned_enc(
        n_channels=2, ref_ch=[0,1], cond_vector_size=192, mask_ch=[0], cond_vector_ch=1, encoder_sample_rate=16000)
    

    e = torch.randn(1, 2, 3*16000)  # enrollment signal with 2 channels, each 3 seconds long
    x = torch.randn(1, 2, 3*16000)  # input signal with 2 channels, each 3 seconds long

    y = model(x, e)
    print(y.shape)

    torchinfo.summary(model, input_size=(x.shape, e.shape))