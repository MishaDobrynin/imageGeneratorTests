from __future__ import annotations  #forward annotations are mostly for readability

import torch  #tensor math + automatic gradients
from torch import nn  #neural-network layers live here

#the network begins from 64 random values and produces a small rgb image.
LATENT_DIM = 64
IMAGE_CHANNELS = 3
IMAGE_SIZE = 64
#keep channel counts modest for cpu training on a 4 gb raspberry pi 4.
BASE_CHANNELS = 16

#give conv and batchnorm layers sensible starting weights instead of leaving defaults in place
def _initialize_weights(module: nn.Module) -> None:
    #dcgans generally behave better when convolution weights start near zero (quote stackoverflow)
    #technically prevents the neurons from calculating the same results
    #random seeding at start fixes this
    name = module.__class__.__name__
    if "Conv" in name: #conv2d
        #extracts spacial features ie edges
        #small random weights should let filters "learn" different jobs
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm" in name: #batchnorm2d
        #normalize data (0 to 1 then shift)
        #batch norm starts close to an identity scale and zero offset
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)

#turn a small random vector into a 64 by 64 rgb image
class Generator(nn.Module):
    #stack upsampling layers until the output reaches the target image size
    def __init__(self, latent_dim: int = LATENT_DIM, base_channels: int = BASE_CHANNELS):
        super().__init__()
        #sequential keeps the layer order readable and lets pytorch call them in order
        #instead of having to call .forward and passing outputs manually
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),  #4x4
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            #every block doubles the spatial size and gradually uses fewer feature channels
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),  #8x8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),  #16x16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),  #32x32
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, IMAGE_CHANNELS, 4, 2, 1, bias=True),  #64x64 rgb
            #tanh caps generated pixels at the  minus-one to one range used by the training loader
            nn.Tanh(),
        )
        self.apply(_initialize_weights)
        #pass sub-layer into _initialize_weights

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        #shape is normally [batch, 64, 1, 1]; output is [batch, 3, 64, 64] (image structure)
        return self.network(noise)

#compress an image into one score used to judge real versus generated examples
#letting the neural network "learn" what is good vs bad
class Discriminator(nn.Module):
    #stack downsampling layers so the image becomes smaller at every stage
    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__()
        #this side compresses an image to one score
        #no sigmoid since bcewithlogitsloss includes it
        #and relu is a lot faster so might as well shave off time
        self.network = nn.Sequential( #sequential is as previously
            nn.Conv2d(IMAGE_CHANNELS, base_channels, 4, 2, 1, bias=True),  #32x32
            #leaky relu leaves a small gradient when activations go negative
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),  #16x16
            nn.BatchNorm2d(base_channels * 2),
            #leaky relu leaves a small gradient when activations go negative (important)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),  #8x8
            nn.BatchNorm2d(base_channels * 4),
            #leaky relu leaves a small gradient when activations go negative (just to re-iterate)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),  #4x4
            nn.BatchNorm2d(base_channels * 8),
            #leaky relu leaves a small gradient when activations go negative (i hope you get the point)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=True),  #one raw real/fake score per image

            #relu is rectified linear unit i.e. max(0,x)
            #leaky relu means every once in a while (20%) negative num goes through
            #i.e. error signals can actually go backwards as theyre corrected
        )
        self.apply(_initialize_weights)

    #run the image through the discriminator and return its score
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        #view(-1) collapses 4d tensor into just [Batch]
        return self.network(image).view(-1)