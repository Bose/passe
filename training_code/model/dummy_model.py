import torch


class DummyModel(torch.nn.Module):
    """
    A dummy model that does nothing but return the input tensor.

    Args:
        reference_channel (int): the channel to return
    """
    def __init__(self, reference_channel: int=0, **kwargs):
        self.reference_channel = reference_channel
        super().__init__(**kwargs)

    def forward(self, x, extra=None):
        return x[:, self.reference_channel, :]