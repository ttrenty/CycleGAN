import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape) # unpacking the tuple self.img_shape
        return img
    
if __name__ == '__main__':
    # Creating an instance of the Generator
    img_shape = (1, 28, 28)  # Example shape
    latent_dim = 100  # Example latent dimension size
    generator = Generator(img_shape, latent_dim)

    # Counting the number of parameters
    num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print("Number of parameters in the Generator:", num_params)