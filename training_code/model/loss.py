import torch

def l1TimeSTFTLoss(output, target):
    """
    Combined time and STFT domain L1 loss function.
    Assumes input signals of shape (batch, 1, samples) - so a batch of one-channel signals.
    """

    # STFT setup 
    nfft=512
    hop=256
    win = torch.hann_window(nfft).to(output.device)

    # value for clamping the magnitudes
    eps = 1e-8

    # remove the channel dimension (signals should be single-channel)
    output = torch.squeeze(output, dim=1)
    target = torch.squeeze(target, dim=1)

    # Disable autocast for complex operations
    with torch.autocast(device_type='cuda', enabled=False):

        # compute magnitudes ()
        output_mag = torch.stft(output, n_fft=nfft, hop_length=hop, win_length=nfft, 
                                window=win, return_complex=True).abs().clamp(min=eps) 
        target_mag = torch.stft(target, n_fft=nfft, hop_length=hop, win_length=nfft, 
                                window=win, return_complex=True).abs().clamp(min=eps) 

    # compute the L1 loss of the time-domain signals  and the STFT magnitudes
    time_loss_val = torch.nn.functional.l1_loss(output+1e-12, target, reduction="mean")
    stft_mag_loss_val = torch.nn.functional.l1_loss(output_mag+1e-12, target_mag, reduction="mean")

    # sort components for individual logging
    parts_loss = {'l1_time': time_loss_val, 'l1_stft_mag': stft_mag_loss_val}

    # add loss components and return
    return time_loss_val + stft_mag_loss_val, parts_loss