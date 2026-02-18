use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv2d_no_bias, conv_transpose2d, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig,
    ConvTranspose2d, ConvTranspose2dConfig, Module, ModuleT, VarBuilder,
};

#[derive(Debug)]
pub struct ConvBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ConvBlock {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(in_channels, out_channels, 3, conv_cfg, vb.pp("conv1"))?;
        let bn1 = batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn1"))?;
        let conv2 = conv2d_no_bias(out_channels, out_channels, 3, conv_cfg, vb.pp("conv2"))?;
        let bn2 = batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn2"))?;
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward_t(&x, train)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, train)?;
        x.relu()
    }
}

#[derive(Debug)]
pub struct StereoUNet {
    enc1: ConvBlock,
    enc2: ConvBlock,
    enc3: ConvBlock,
    enc4: ConvBlock,
    bottleneck: ConvBlock,
    up4: ConvTranspose2d,
    dec4: ConvBlock,
    up3: ConvTranspose2d,
    dec3: ConvBlock,
    up2: ConvTranspose2d,
    dec2: ConvBlock,
    up1: ConvTranspose2d,
    dec1: ConvBlock,
    disparity_head: Conv2d,
    logvar_head: Conv2d,
}

impl StereoUNet {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        base_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let c1 = base_channels;
        let c2 = c1 * 2;
        let c3 = c2 * 2;
        let c4 = c3 * 2;
        let c5 = c4 * 2;

        let enc1 = ConvBlock::new(in_channels, c1, vb.pp("enc1"))?;
        let enc2 = ConvBlock::new(c1, c2, vb.pp("enc2"))?;
        let enc3 = ConvBlock::new(c2, c3, vb.pp("enc3"))?;
        let enc4 = ConvBlock::new(c3, c4, vb.pp("enc4"))?;
        let bottleneck = ConvBlock::new(c4, c5, vb.pp("bottleneck"))?;

        let up_cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let up4 = conv_transpose2d(c5, c4, 2, up_cfg, vb.pp("up4"))?;
        let dec4 = ConvBlock::new(c4 + c4, c4, vb.pp("dec4"))?;
        let up3 = conv_transpose2d(c4, c3, 2, up_cfg, vb.pp("up3"))?;
        let dec3 = ConvBlock::new(c3 + c3, c3, vb.pp("dec3"))?;
        let up2 = conv_transpose2d(c3, c2, 2, up_cfg, vb.pp("up2"))?;
        let dec2 = ConvBlock::new(c2 + c2, c2, vb.pp("dec2"))?;
        let up1 = conv_transpose2d(c2, c1, 2, up_cfg, vb.pp("up1"))?;
        let dec1 = ConvBlock::new(c1 + c1, c1, vb.pp("dec1"))?;

        let disparity_head = candle_nn::conv2d(
            c1,
            out_channels,
            1,
            Default::default(),
            vb.pp("disparity_head"),
        )?;
        let logvar_head = candle_nn::conv2d(c1, 1, 1, Default::default(), vb.pp("logvar_head"))?;

        Ok(Self {
            enc1,
            enc2,
            enc3,
            enc4,
            bottleneck,
            up4,
            dec4,
            up3,
            dec3,
            up2,
            dec2,
            up1,
            dec1,
            disparity_head,
            logvar_head,
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let s1 = self.enc1.forward_t(x, train)?;
        let s2 = self.enc2.forward_t(&s1.max_pool2d(2)?, train)?;
        let s3 = self.enc3.forward_t(&s2.max_pool2d(2)?, train)?;
        let s4 = self.enc4.forward_t(&s3.max_pool2d(2)?, train)?;
        let b = self.bottleneck.forward_t(&s4.max_pool2d(2)?, train)?;

        let d4 = self.up4.forward(&b)?;
        let d4 = Tensor::cat(&[&d4, &s4], 1)?;
        let d4 = self.dec4.forward_t(&d4, train)?;

        let d3 = self.up3.forward(&d4)?;
        let d3 = Tensor::cat(&[&d3, &s3], 1)?;
        let d3 = self.dec3.forward_t(&d3, train)?;

        let d2 = self.up2.forward(&d3)?;
        let d2 = Tensor::cat(&[&d2, &s2], 1)?;
        let d2 = self.dec2.forward_t(&d2, train)?;

        let d1 = self.up1.forward(&d2)?;
        let d1 = Tensor::cat(&[&d1, &s1], 1)?;
        let d1 = self.dec1.forward_t(&d1, train)?;

        let disparity = softplus(&self.disparity_head.forward(&d1)?)?;
        let logvar = self.logvar_head.forward(&d1)?.clamp(-6f32, 3f32)?;
        Ok((disparity, logvar))
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // Stable softplus: max(x, 0) + log(1 + exp(-abs(x))).
    let max_part = x.maximum(0f32)?;
    let log_part = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    max_part.broadcast_add(&log_part)
}
