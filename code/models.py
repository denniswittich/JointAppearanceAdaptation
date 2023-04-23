import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ssl
import segmentation_models_pytorch as smp
import config
from tools import may_print as mprint


# ============================================================================= High level model initializer

def model_from_config(net_type: str, cf: config.Config):
    """Setup a network of type 'net_type' based on the configuration object.

    :param net_type: String code for the network type
    :param cf: Configuration object
    """
    if net_type == 'seg':
        type = cf.SEG_MODEL.TYPE
        if type == 'unet':
            return UnetSMP(cf)
    elif net_type == 'ap_ad':
        type = cf.AP_AD_MODEL.TYPE
        if type == 'res_fcn':
            return ResFCN(cf.AP_AD_MODEL)
    elif net_type == 'ap_dis':
        return ImgDiscriminator(cf.AP_DIS_MODEL)

    elif net_type == 'aux_gen':
        type = cf.AUX_GEN_MODEL.TYPE
        if type == 'tconf':
            return AuxGenEncoder(cf.AUX_GEN_MODEL)

    elif net_type == 'rp_dis':
        return RepDiscriminator(cf.RP_DIS_MODEL)

    raise NotImplementedError(f"Creating model for {net_type} failed")


# ============================================================================= U-NET FROM SMP LIBRARY

class UnetSMP(nn.Module):

    def __init__(self, cf: config.Config):
        """Initialize U-Net using SMP library. Uses config cf.SEG_MODEL

        Uses Semantic segmentation library. The network returns class logits.
        (pip install git+https://github.com/qubvel/segmentation_models.pytorch)
        :param cf: Configuration object
        """

        super(UnetSMP, self).__init__()

        # Fixing URLError when loading pretrained weights!
        ssl._create_default_https_context = ssl._create_unverified_context

        n_out, n_in = cf.SEG_MODEL.OUT_CHN, cf.SEG_MODEL.IN_CHN
        udep = cf.SEG_MODEL.UNET.DEPTH

        if hasattr(cf.DA, "DO_REPA") and cf.DA.DO_REPA:
            self.rep_layer = cf.DA.REP_LAYER
            if self.rep_layer == 'encoder' or self.rep_layer == 'tsai2':
                self.encoder_stage = cf.DA.REP_ENCODER_STAGE

        assert 3 <= udep <= 5, f"Invalid U-Net encoder depth {udep}. Should be 3 >= depth >= 5"
        mprint(f'Initializing UNET {n_in} -> {n_out} [depth={udep}]', 2, cf.VRBS, True)

        if cf.SEG_MODEL.UNET.PRETRAINED:
            mprint(' (using pretrained imagenet weights)', 2, cf.VRBS)
            encoder_weights = 'imagenet'
        else:
            mprint(' (training from scratch)', 2, cf.VRBS)
            encoder_weights = None

        self.unet = smp.Unet(cf.SEG_MODEL.UNET.BACKBONE, in_channels=n_in, classes=n_out, encoder_depth=udep,
                             decoder_channels=[256, 128, 64, 32, 16][-udep:], encoder_weights=encoder_weights)

        if udep <= 4:  # TODO: more parameters can be removed for udep = 3!
            self.unet.encoder.block12 = None
            self.unet.encoder.conv3 = None
            self.unet.encoder.bn3 = None
            self.unet.encoder.conv4 = None
            self.unet.encoder.bn4 = None

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        return self.unet(x)

    def getReps(self, x):
        """Returns activation maps (which are interpreted as latent representations for representation transfer).

        :param x: The data for which the activation maps should be returned
        """

        if self.rep_layer == 'encoder':
            features = self.unet.encoder(x)[self.encoder_stage]  # 3 ->  Bx32x32x32
            return features
        elif self.rep_layer == 'decoder':
            features = self.unet.encoder(x)
            decoder_output = self.unet.decoder(*features)
            return decoder_output  # BxHxWx16
        elif self.rep_layer == 'tsai2':
            features = self.unet.encoder(x)
            decoder_output = self.unet.decoder(*features)
            return features[self.encoder_stage], decoder_output  # BxHxWx16
        elif self.rep_layer == 'ent_maps':  # used for Adversarial entropy minimisation
            logits = self.unet(x)
            probabilities = torch.softmax(logits, dim=-1)
            pre_ent_maps = probabilities * torch.log(probabilities + 1e-5) * -1.0
            return pre_ent_maps
        else:
            raise NotImplementedError(f'The value "{self.rep_layer}" is not supported.')


# ============================================================================= APPEARANCE ADAPTATION

class Down4(nn.Module):
    def __init__(self, in_c, stride, out_c, batch_norm=False):
        """Down-sampling with strided convolution.

        Uses replicate padding to prevent artefacts.
        :param in_c: Number of input channels
        :param stride: Step width of kernel -> downsampling factor
        :param out_c: Number of output channels
        :param batch_norm: Use batch normalization?
        """
        super(Down4, self).__init__()
        if batch_norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, (6, 6), stride, 1, padding_mode='replicate'),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, (6, 6), stride, 1, padding_mode='replicate'),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        return self.conv(x)


class ResBlock(nn.Module):

    def __init__(self, ch_io):
        """Residual block of two convolutional layers.

        Uses replicate padding to prevent artefacts. Activation function (relu) is applied before adding the residual!
        :param ch_io: Number of channels in input and output
        """
        super(ResBlock, self).__init__()
        ch_p = ch_io // 4
        pad = 'replicate'

        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_io, ch_p, (3, 3), padding=1, padding_mode=pad), nn.ReLU(inplace=True),
            nn.Conv2d(ch_p, ch_io, (3, 3), padding=1, padding_mode=pad), nn.ReLU(inplace=True))

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        return x + self.double_conv(x)


class ResFCN(nn.Module):
    def __init__(self, cf: config.Config.AP_AD_MODEL, batch_norm=False):
        """Fully convolutional network with residual blocks.

        Used for appearance adaptation (image-to-image translation).
        :param cf: Configuration sub-object for appearance adaptation model.
        :param batch_norm: Whether to use batch normalization layers.
        """
        super(ResFCN, self).__init__()
        fn = cf.RNET.NUM_FEAT
        n_blcs = cf.RNET.NUM_BLOCKS
        self.dr_rate = cf.RNET.DROPRATE

        assert cf.RNET.SCALE == 4, 'Currently only scale 4 is implemented.'
        self.down = Down4(cf.IN_CHN, cf.RNET.SCALE, fn, batch_norm=batch_norm)

        if batch_norm:
            self.mid = nn.Sequential(*[ResBlock(fn) for _ in range(n_blcs)], nn.BatchNorm2d(fn))
            self.up = nn.Sequential(nn.ConvTranspose2d(fn, fn // 2, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.BatchNorm2d(fn // 2), nn.ReLU(True))
        else:
            self.mid = nn.Sequential(*[ResBlock(fn) for _ in range(n_blcs)])
            self.up = nn.Sequential(nn.ConvTranspose2d(fn, fn // 2, kernel_size=4, stride=2, padding=1, bias=True),
                                    nn.ReLU(True))

        self.out = nn.ConvTranspose2d(fn // 2, cf.OUT_CHN, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        xD = self.down(x)
        xM = self.mid(xD)
        if self.dr_rate:
            xM = nn.functional.dropout(xM, inplace=True, p=self.dr_rate)
        xT = self.up(xM)
        xU = self.out(xT)
        return xU

    def process(self, xD):
        """Process an activation by the segmentation head.

        Used in the context of weight-sharing.
        :param xD: Activation map to feed to the segmentation head.
        """
        xM = self.mid(xD)
        xT = self.up(xM)
        xU = self.out(xT)
        return xU


# ============================================================================= AUXILIARY GENERATOR

class AuxGenEncoder(nn.Module):
    def __init__(self, cf):
        """Used to generate synthetic representations which are processed by the decoder of the ap.ad. network).

        See also ResFCN.process.
        :param cf: Configuration object.
        """

        def layer(inf, out_c, ks, st, pa, up, spn, dil=1, bias=False, batch_norm=True):
            SpNo = nn.utils.spectral_norm

            if up:
                C = nn.ConvTranspose2d(inf, out_c, kernel_size=ks, stride=st, padding=pa, dilation=dil, bias=bias)
            else:
                C = nn.Conv2d(inf, out_c, kernel_size=ks, stride=st, padding=pa, dilation=dil, bias=bias)

            if spn:
                C = SpNo(C)

            A = nn.ReLU(inplace=True)
            if batch_norm:
                BN = nn.BatchNorm2d(out_c)
                return C, BN, A

            return C, A

        super(AuxGenEncoder, self).__init__()
        self.dr_rate = cf.DROPRATE
        spn = False
        p = cf.START_SIZE

        self.backconf = nn.Sequential(
            *layer(cf.IN_CHN, 256, p, 1, 0, True, spn),  # --- 1 -> p
            *layer(256, 512, 3, 1, 1, False, spn),  # -------- p -> p
        )
        self.to16 = nn.Sequential(
            *layer(512, 256, 4, 2, 1, True, spn),  # --------- p -> 2p
            *layer(256, 256, 3, 1, 1, False, spn),  # -------- 2p -> 2p
        )
        self.to32 = nn.Sequential(
            *layer(256, 256, 4, 2, 1, True, spn),  # --------- 2p -> 4p
            *layer(256, 256, 3, 1, 1, False, spn),  # -------- 4p -> 4p
        )
        self.to64 = nn.Sequential(
            *layer(256, 256, 4, 2, 1, True, spn),  # --------- 4p -> 4p
            *layer(256, cf.OUT_CHN, 3, 1, 1, False, spn=False, bias=True, batch_norm=False),  # - 4p -> 4p
        )

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        x8 = self.backconf(x)
        x16 = self.to16(x8)
        if self.dr_rate: x16 = nn.functional.dropout(x16, p=self.dr_rate)
        x32 = self.to32(x16)
        x64 = self.to64(x32)
        return x64


# ============================================================================= DISCRIMINATORS

class ImgDiscriminator(nn.Module):
    def __init__(self, cf: config.Config.AP_DIS_MODEL):
        """ Set up the image (appearance) discriminator.

        The network returns probabilistic class scores.
        Possible types are 'cnn', 'patch-gan', 'benjdira'. All have a receptive field of 70 x 70 px.
        'patch-gan' corresponds to the discriminator, proposed in (Isola et al., 2017)
        'benjdira' modifies the patch-gan discriminator by a dropout layer (Benjdira et al., 2019)
        'cnn' modifies the patch-gan discriminator by spectral normalization (Used in the thesis)
        :param cf: Configuration object.
        """
        super(ImgDiscriminator, self).__init__()
        SpNo = nn.utils.spectral_norm
        self.shift_input = cf.SHIFT

        if cf.TYPE == 'cnn':  # Rec Field = 70
            self.fw = nn.Sequential(
                nn.Conv2d(cf.IN_CHN, 64, kernel_size=4, stride=2, padding=0, bias=True),  # 126 / 62
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                SpNo(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, bias=True)),  # 62 / 30
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                SpNo(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0, bias=True)),  # 30 / 14
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                SpNo(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=True)),  # 27 / 11
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                SpNo(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=True)),  # 24 / 8
                nn.Sigmoid(),
            )
        elif cf.TYPE in ['patch-gan', 'benjdira']:  # RF = 70
            print("Creating 'patch-gan' discriminator")
            self.fw = nn.Sequential(
                nn.Conv2d(cf.IN_CHN, 64, kernel_size=4, stride=2, padding=0, bias=True),  # 126 / 62
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Dropout2d(p=0.1) if cf.TYPE == 'benjdira' else nn.Identity(),

                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, bias=False),  # 62 / 30
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0, bias=False),  # 30 / 14
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=True),  # 27 / 11
                nn.LeakyReLU(inplace=True, negative_slope=0.2),

                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=True),  # 24 / 8
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError(f'DISCRIMINATOR {cf.TYPE} is not implemented!')

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        if self.shift_input:
            rx, ry = np.random.randint(0, 4, 2)
            return self.fw(x[:, :, rx:, ry:])
        else:
            return self.fw(x)


class RepDiscriminator(nn.Module):
    def __init__(self, cf: config.Config.RP_DIS_MODEL):
        """ Set up the representation discriminator.

        The network returns probabilistic class scores.
        Possible types are 'cnn', 'mlp', 'mlp2', 'ran'
        'cnn': Patch-gan discriminator with spectral normalization
        'mlp': Implementation of an mlp as discriminator
        'ran': Discriminator from (Zhang et al., 2018a)
        'advent'/'tsai': Discriminator from (Vu et al., 2019) / (Tsai et al., 2018)
        'tsai2': Dual discriminator according to (Tsai et al., 2018) (representation transfer at two layers)
        :param cf: Configuration object for representation discriminator.
        """
        super(RepDiscriminator, self).__init__()
        SpNo = nn.utils.spectral_norm
        self.C = cf
        fs = cf.NUM_F_START

        if cf.TYPE == 'cnn':
            self.fw = nn.Sequential(
                nn.Conv2d(cf.IN_CHN, fs, kernel_size=4, stride=2, padding=0, bias=True),  # 126 / 62
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs, fs * 2, kernel_size=4, stride=2, padding=0, bias=True)),  # 62 / 30
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs * 2, fs * 4, kernel_size=4, stride=1, padding=0, bias=True)),  # 30 / 14
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs * 4, 1, kernel_size=4, stride=1, padding=0, bias=True)),  # 30 / 14
                nn.Sigmoid(),
            )
        elif cf.TYPE == 'mlp':
            self.fw = nn.Sequential(
                nn.Conv2d(cf.IN_CHN, fs, kernel_size=1, bias=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs, fs * 2, kernel_size=1, bias=True)),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs * 2, fs * 4, kernel_size=1, bias=True)),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                SpNo(nn.Conv2d(fs * 4, 1, kernel_size=1, bias=True)),
                nn.Sigmoid(),
            )
        elif cf.TYPE == 'ran':
            fs = 128
            self.c1 = nn.Conv2d(cf.IN_CHN, fs, 3, padding=1, dilation=1)
            self.c2 = nn.Conv2d(cf.IN_CHN, fs, 3, padding=3, dilation=2)
            self.c3 = nn.Conv2d(cf.IN_CHN, fs, 3, padding=5, dilation=3)
            self.c4 = nn.Conv2d(cf.IN_CHN, fs, 3, padding=7, dilation=4)

            self.out = nn.Sequential(
                nn.Conv2d(fs * 4, 1, 1),
                nn.Sigmoid(),
            )
            self.fw = self.fw_ran
        elif cf.TYPE == 'advent' or cf.TYPE == 'tsai':
            ndf = fs
            self.fw = nn.Sequential(
                nn.Conv2d(cf.IN_CHN, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        elif cf.TYPE == 'tsai2':
            ndf = fs
            self.d1 = nn.Sequential(
                nn.Conv2d(cf.IN_CHN[0], ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
            self.d2 = nn.Sequential(
                nn.Conv2d(cf.IN_CHN[1], ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
            self.fw = self.fw_tsai2
        else:
            raise NotImplementedError(f'Representation discriminator {cf.TYPE} is not implemented!')

    def fw_tsai2(self, x):
        return [self.d1(x[0]), self.d2(x[1])]

    def fw_ran(self, x):
        catted = torch.cat((self.c1(x), self.c2(x), self.c3(x), self.c4(x)), 1)
        return self.out(catted)

    def forward(self, x, random_shift=False):
        """Forward pass (cf. nn.Module)"""
        if random_shift:
            rx, ry = np.random.randint(0, 4, 2)
            return self.fw(x[:, :, rx:, ry:])
        else:
            return self.fw(x)
