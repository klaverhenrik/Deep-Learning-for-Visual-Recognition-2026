# Lecture 9 — Generative Models

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes cover the three main families of deep generative models — autoencoders, variational autoencoders (VAEs), and GANs — explaining the motivation for each, the mathematical machinery, and the practical failure modes. A comparison table at the end places them alongside diffusion models (covered in Lecture 13).

---

## 1  Unsupervised Learning and Generative Models

A **discriminative model** learns $P(y \mid \mathbf{x})$ — the probability of a label given an input. A **generative model** learns $P(\mathbf{x})$ — the probability distribution of the data itself. This distinction matters because:

- Generative models can **create new data** by sampling from the learned distribution
- They can learn **useful representations** without labels (unsupervised)
- They model the full data distribution, not just the boundary between classes

The fundamental challenge: real image distributions are extremely high-dimensional and complex. A 64×64 RGB image lives in a $64 \times 64 \times 3 = 12{,}288$-dimensional space. Modelling this directly is intractable.

---

## 2  Autoencoders

### 2.1  Architecture

An autoencoder learns to compress data into a low-dimensional latent representation (encoding), then reconstruct it (decoding):

$$\mathbf{z} = f_\text{enc}(\mathbf{x}), \qquad \hat{\mathbf{x}} = f_\text{dec}(\mathbf{z})$$

The **bottleneck** forces the network to retain only the most important information. Training minimises reconstruction loss:

$$J = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \quad \text{(MSE)}$$

### 2.2  Variants

- **Undercomplete AE**: $\dim(\mathbf{z}) \ll \dim(\mathbf{x})$ — standard dimensionality reduction
- **Denoising AE**: input is corrupted; model learns to recover clean signal — more robust features
- **Sparse AE**: regularise latent representation to be sparse — $J_\text{reg} = J + \lambda \|\mathbf{z}\|_1$

### 2.3  The Generation Problem

Standard autoencoders are not generative models: the latent space has **no guaranteed structure**. Different classes may cluster in disjoint regions with gaps between them. Sampling a random point in latent space and decoding it often produces garbage.

```python
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),   # 64→32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 32→16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(), # 16→8
            nn.Flatten(),
            nn.Linear(128*8*8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8), nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),  nn.Sigmoid(),
        )

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

ae = ConvAutoencoder(latent_dim=128)
x  = torch.randn(4, 3, 64, 64)
x_hat, z = ae(x)
print(f'Input: {x.shape} → Latent: {z.shape} → Reconstructed: {x_hat.shape}')
```

---

## 3  Variational Autoencoders (VAEs)

### 3.1  The Key Idea

The VAE (Kingma & Welling, 2013) imposes a **structured latent space** by training the encoder to produce a probability distribution over $\mathbf{z}$, not a single point. Specifically, the encoder outputs the mean $\boldsymbol{\mu}$ and log-variance $\log\boldsymbol{\sigma}^2$ of a Gaussian:

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2(\mathbf{x})))$$

The KL divergence term in the loss pushes this distribution towards a standard normal $\mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$J_\text{VAE} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{reconstruction}} + \underbrace{D_\text{KL}(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0},\mathbf{I}))}_{\text{regularisation}}$$

### 3.2  The Reparameterisation Trick

We need to sample $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ during the forward pass, but sampling is not differentiable. The reparameterisation trick rewrites sampling as:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now $\boldsymbol{\epsilon}$ is an external source of randomness independent of the parameters, so gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ normally.

### 3.3  KL Divergence Closed Form

For a Gaussian encoder $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ against standard normal $\mathcal{N}(\mathbf{0},\mathbf{I})$, the KL has a closed form:

$$D_\text{KL} = -\frac{1}{2} \sum_{j} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 3.4  Generation

After training, generate new images by sampling $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and decoding. The KL regularisation ensures the latent space is smooth: nearby points in $\mathbf{z}$ decode to similar images, and the space has no gaps.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        enc_out = 64 * 7 * 7
        self.fc_mu     = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, enc_out), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1,  4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps   # z = mu + sigma * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z     = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return recon + beta * kl

vae = VAE(latent_dim=64)
x   = torch.rand(8, 1, 28, 28)   # MNIST-like
x_hat, mu, logvar = vae(x)
loss = vae_loss(x, x_hat, mu, logvar)
print(f'Loss: {loss.item():.2f}')

# Generate new images
with torch.no_grad():
    z     = torch.randn(16, 64)     # sample from N(0,I)
    imgs  = vae.decode(z)           # decode to image space
    print(f'Generated: {imgs.shape}')  # (16, 1, 28, 28)
```

---

## 4  Generative Adversarial Networks (GANs)

### 4.1  The Minimax Game

GANs (Goodfellow et al., 2014) train two networks in opposition:

- **Generator** $G$: maps noise $\mathbf{z} \sim P_z$ to fake images $G(\mathbf{z})$, trying to fool the discriminator
- **Discriminator** $D$: classifies images as real or fake, trying to detect the generator's fakes

The minimax objective:

$$\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim P_\text{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim P_z}[\log(1 - D(G(\mathbf{z})))]$$

At the Nash equilibrium, $G$ produces samples indistinguishable from real data and $D$ outputs 0.5 everywhere.

### 4.2  Training

Training requires alternating updates — separate optimisers for $G$ and $D$:

1. **Update $D$**: sample real images and fake images ($G(\mathbf{z})$); minimise $-[\log D(\mathbf{x}) + \log(1-D(G(\mathbf{z})))]$
2. **Update $G$**: generate fake images; maximise $\log D(G(\mathbf{z}))$ (equivalently, minimise $-\log D(G(\mathbf{z}))$ — the non-saturating variant)

### 4.3  Failure Modes

- **Mode collapse**: $G$ learns to generate a few convincing examples but ignores most of the data distribution
- **Vanishing gradient for $G$**: if $D$ becomes too good early on, $\log(1-D(G(\mathbf{z}))) \approx 0$ and gradients to $G$ vanish

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, img_size=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),        nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),       nn.LeakyReLU(0.2),
            nn.Linear(1024, img_channels * img_size**2), nn.Tanh(),
        )
        self.img_shape = (img_channels, img_size, img_size)

    def forward(self, z):
        return self.net(z).view(-1, *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_channels * img_size**2, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),                        nn.LeakyReLU(0.2),
            nn.Linear(256, 1),                          nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img.view(img.size(0), -1))

G = Generator();  D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()

real = torch.rand(32, 1, 28, 28)      # simulated real batch
z    = torch.randn(32, 100)

# ── Train D ──────────────────────────────────────────────────────────
opt_D.zero_grad()
real_loss = loss_fn(D(real), torch.ones(32, 1))
fake_loss = loss_fn(D(G(z).detach()), torch.zeros(32, 1))  # detach: no G gradients
d_loss = real_loss + fake_loss
d_loss.backward();  opt_D.step()

# ── Train G (non-saturating: maximise log D(G(z))) ───────────────────
opt_G.zero_grad()
g_loss = loss_fn(D(G(z)), torch.ones(32, 1))  # want D to say "real"
g_loss.backward();  opt_G.step()
print(f'D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}')
```

---

## 5  Comparison

| Property | Autoencoder | VAE | GAN | Diffusion |
|---|---|---|---|---|
| Training | Stable (MSE) | Stable (ELBO) | Unstable (adversarial) | Stable (MSE on noise) |
| Sample quality | Poor (blurry) | Moderate (blurry) | High | Highest |
| Latent space | Unstructured | Smooth, Gaussian | No explicit latent | No explicit latent |
| Generation | Interpolation only | Sample $\mathcal{N}(\mathbf{0},\mathbf{I})$ | Sample $\mathcal{N}(\mathbf{0},\mathbf{I})$ | Iterative denoising |
| Speed | Fast | Fast | Fast | Slow ($T$ steps) |
| Key failure | Blurry recon | Blurry outputs | Mode collapse | Slow inference |

The evolution from AE → VAE → GAN → Diffusion models is a story of progressively better sample quality at the cost of training stability (GAN) or inference speed (Diffusion). Diffusion models (Lecture 13) resolve the training instability of GANs while achieving superior quality.

## References

- Kingma, D. & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR 2014.
- Goodfellow, I. et al. (2014). Generative Adversarial Networks. NeurIPS.
- Radford, A. et al. (2015). Unsupervised Representation Learning with DCGAN. ICLR 2016.
