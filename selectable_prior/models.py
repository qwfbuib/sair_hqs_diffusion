import torch
from sair_unfold.model import UNet
from sair_unfold.prior import DiffusionPrior
from sair_unfold.unfolding import HQSUnfold
from .ss2d import Mamba2DBlock


def architecture(backbone):
    if backbone == 'unet':
        return dict(backbone='unet', version='legacy_unet_ve_v1', scan_backend=None)
    if backbone == 'mamba':
        return dict(backbone='mamba', version='hybrid_ss2d_ve_v1',
                    scan_backend=Mamba2DBlock.backend,
                    replacements=['encoder.2.1', 'middle.1'], d_state=16,
                    expand=2, d_conv=3, chunk_size=32, time_conditioning=True)
    raise ValueError('prior.backbone must be unet or mamba')


class MambaDenoiser(UNet):
    """The original topology with two low-resolution ResBlocks replaced."""
    def __init__(self, base_channels=32, channel_mult=(1, 2, 4, 4)):
        if len(channel_mult) != 4:
            raise ValueError('Mamba hybrid v1 requires exactly four resolution levels')
        super().__init__(base_channels, channel_mult)
        opts=dict(time_width=base_channels*4, d_state=16, expand=2, d_conv=3,
                  chunk_size=32, time_conditioning=True, checkpoint_scan=True)
        self.encoder[2][1]=Mamba2DBlock(base_channels*channel_mult[2], **opts)
        self.middle[1]=Mamba2DBlock(base_channels*channel_mult[-1], **opts)


class SelectablePrior(DiffusionPrior):
    """Inherit exactly the existing VE epsilon/denoising/loss definitions."""
    def __init__(self, config, backbone='unet'):
        torch.nn.Module.__init__(self)
        architecture(backbone)
        self.config=config
        self.backbone=backbone
        self.net=(UNet if backbone=='unet' else MambaDenoiser)(
            config.base_channels, config.channel_mult)


class SelectableHQS(HQSUnfold):
    def __init__(self, config, backbone='unet', projection='soft'):
        # Build only the shared phase parameters, then install the chosen prior.
        super().__init__(config, prior_kind='none', projection=projection)
        self.prior_kind='diffusion'
        self.backbone=backbone
        self.prior=SelectablePrior(config, backbone)
