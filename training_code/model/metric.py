import torch
import torchmetrics


def pesq_wb(output, target):

    results_per_example = torchmetrics.functional.audio.pesq.perceptual_evaluation_speech_quality(
        preds=output, target=target, fs=16000, mode='wb').float()
    
    return results_per_example


def si_sdr(output, target):
    """
    Object-based, used in training, returns batch-averaged result
    """
    sisdr = torchmetrics.audio.ScaleInvariantSignalDistortionRatio().to(target.device)

    return sisdr(preds=output, target=target).float()


def si_sdr_per_example(output, target):
    """
    Function-based, used in training, returns result per example
    """
    results_per_example = torchmetrics.functional.audio.scale_invariant_signal_noise_ratio(
        preds=output, target=target).float()
    
    return torch.flatten(results_per_example)


def estoi(output, target):
    results_per_example = torchmetrics.functional.audio.short_time_objective_intelligibility(
        preds=output, target=target, fs=16000, extended=True).float()
    
    return results_per_example