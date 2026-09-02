# Lecture 9
# Generative Models

*Deep Learning for Visual Recognition · Aarhus University*

These notes tell a single coherent story: starting from the question 'how do we generate new images?', building a standard autoencoder, identifying its failure modes, extending it to a VAE to fix the latent space, and then motivating GANs to fix the blurriness. Each architectural leap is motivated by the shortcomings of the previous one.

---

## 1  Unsupervised Learning and Generative Models

### 1.1  Supervised vs Unsupervised Learning

All models covered so far — classifiers, object detectors, segmentation networks — are trained with labelled data: every training example comes with a ground-truth label or bounding box or mask. This is supervised learning. Labels are expensive and their scarcity is one of the main bottlenecks in deep learning.

Unsupervised learning uses data without labels, with the goal of learning the underlying structure of the data distribution itself. Useful applications include dimensionality reduction (compressing high-dimensional data into a low-dimensional representation that retains the important structure), anomaly detection (things that don't fit the learned structure are anomalies), and — the focus of this lecture — generative modelling: learning to produce new samples that look like they came from the training distribution.

### 1.2  Generative vs Discriminative Models

The distinction between generative and discriminative models is fundamental:

- **Discriminative model**: learns the conditional distribution $p(y \mid \mathbf{x})$ — given an image $\mathbf{x}$, what is the probability of label $y$? It draws a boundary between classes. It can ignore large parts of the data distribution and focus only on the features that distinguish classes.
- **Generative model**: learns the joint distribution $p(\mathbf{x}, y)$ or, in the unsupervised case, $p(\mathbf{x})$ alone — the full distribution of what images look like. It must model how all pixels correlate with each other, which is a far harder problem.

An analogy: a discriminative model tells you whether a digit is a '7' or an '8'. A generative model can write a '7' from scratch — it must understand what a '7' actually looks like, not just how it differs from an '8'.

> **Why is generative modelling hard?** A classifier only needs to learn a decision boundary in pixel space. A generative model must learn that 'boats appear near water', 'eyes are not on foreheads', and the subtle pixel correlations that make images look photorealistic. The space of natural images is a tiny, highly structured manifold inside the vast space of all possible pixel arrays — and the generative model must learn to stay on that manifold.

---

## 2  Autoencoders

### 2.1  The Basic Idea

An autoencoder is a neural network trained to copy its input to its output — but with a bottleneck in the middle that forces it to learn a compressed representation. The network consists of two parts:

- **Encoder** $f$: maps the input $\mathbf{x}$ to a latent (hidden) representation $\mathbf{h} = f(\mathbf{x})$. The latent space has lower dimensionality than $\mathbf{x}$, forcing the encoder to retain only the most important information.
- **Decoder** $g$: maps the latent representation back to the input space: $\hat{\mathbf{x}} = g(\mathbf{h}) = g(f(\mathbf{x}))$. The goal is $\hat{\mathbf{x}} \approx \mathbf{x}$.

Training is entirely unsupervised — no labels are needed. The loss function measures reconstruction quality:

$$J(\mathbf{W}) = \frac{1}{2n} \sum_i \left\| g(f(\mathbf{x}^{(i)})) - \mathbf{x}^{(i)} \right\|^2 \qquad \text{[MSE / L2 loss]}$$

or, for binary inputs, the cross-entropy reconstruction loss. The bottleneck forces the network to learn the most informative features of the data — it cannot simply memorise by copying every pixel through, so it must find structure.

### 2.2  Undercomplete vs Overcomplete Autoencoders

- **Undercomplete AE** ($\dim(\mathbf{h}) < \dim(\mathbf{x})$): The bottleneck is smaller than the input. The network is forced to compress, discovering the most important correlations in the data. Common use: dimensionality reduction, anomaly detection (inputs that reconstruct poorly are likely anomalies).
- **Overcomplete AE** ($\dim(\mathbf{h}) > \dim(\mathbf{x})$): The bottleneck is larger than the input. Without regularisation, the network can learn the trivial identity mapping. Regularisation (sparsity, noise) is required to force learning of meaningful structure.

### 2.3  Deeper Is Better: Stacked Autoencoders

A stacked autoencoder uses multiple hidden layers in both encoder and decoder, e.g. $784 \to 1000 \to 500 \to 250 \to 30$ (bottleneck) $\to 250 \to 500 \to 1000 \to 784$. The extra layers allow the network to learn hierarchical features — similar to how deep classifiers learn edges, then parts, then objects. Empirically, stacked AEs produce much better representations than single-layer ones, and the 2D latent space visualisations show cleaner, better-separated clusters.

### 2.4  Convolutional Autoencoders

For images, replacing fully connected layers with convolutions produces far better results. The encoder uses Conv+ReLU+MaxPool blocks to progressively compress the spatial dimensions while building up channel depth. The decoder mirrors this with transposed convolutions (also called deconvolutions) that upsample the spatial dimensions back to the original resolution.

Transposed convolution (`nn.ConvTranspose2d` in PyTorch) is the learnable alternative to simple bilinear or nearest-neighbour upsampling. It inserts learned values between existing positions, effectively going backwards through a convolution. It has trainable parameters, so the network can learn the optimal upsampling kernel for its task.

```python
import torch
import torch.nn as nn

# ── Standard autoencoder: encoder + bottleneck + decoder ─────────────
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: compresses 1×28×28 → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # → 32×14×14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # → 64×7×7
            nn.ReLU(),
            nn.Flatten(),                               # → 64*7*7 = 3136
            nn.Linear(3136, latent_dim),               # → latent_dim
        )
        # Decoder: expands latent_dim → 1×28×28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),                # → 64×7×7
            nn.ConvTranspose2d(64, 32, 3, stride=2,    # → 32×14×14
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2,     # → 1×28×28
                               padding=1, output_padding=1),
            nn.Sigmoid(),   # pixel values in [0,1]
        )

    def encode(self, x):  return self.encoder(x)
    def decode(self, h):  return self.decoder(h)
    def forward(self, x): return self.decode(self.encode(x))

model = Autoencoder(latent_dim=32)
x = torch.randn(8, 1, 28, 28)   # batch of 8 MNIST images
x_hat = model(x)
print(f'Input:  {x.shape}')      # (8, 1, 28, 28)
print(f'Output: {x_hat.shape}')  # (8, 1, 28, 28)  — same shape

# Loss: MSE between reconstruction and original
loss = nn.MSELoss()(x_hat, x)
print(f'Reconstruction loss: {loss.item():.4f}')

# ── Training loop (unsupervised — no labels needed!) ──────────────────
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):   # abbreviated for illustration
    optimiser.zero_grad()
    x_hat = model(x)
    loss  = nn.MSELoss()(x_hat, x)
    loss.backward()
    optimiser.step()
    print(f'Epoch {epoch+1}  loss: {loss.item():.4f}')

# ── Latent space interpolation ────────────────────────────────────────
model.eval()
with torch.no_grad():
    h1 = model.encode(x[[0]])   # latent vector for image 0
    h2 = model.encode(x[[1]])   # latent vector for image 1

    # Interpolate: α=0 → h1, α=1 → h2
    alphas = torch.linspace(0, 1, 8)
    for alpha in alphas:
        h_interp = (1 - alpha) * h1 + alpha * h2
        x_interp = model.decode(h_interp)   # decode the interpolated latent
        # In practice: visualise x_interp to see smooth transition
```

*Code 1 – A convolutional autoencoder for MNIST. The key architecture choices: strided convolutions for downsampling in the encoder, `ConvTranspose2d` for upsampling in the decoder, and `Sigmoid` on the output to produce pixel values in $[0,1]$. Training is entirely unsupervised — labels are never used.*

### 2.5  Autoencoder Variants

**Denoising Autoencoder (DAE).** A denoising autoencoder is trained to reconstruct a clean input $\mathbf{x}$ from a corrupted version $\tilde{\mathbf{x}}$. The corruption can be random pixel dropout (set random pixels to zero with probability $v$), Gaussian noise, or other noise processes. The key insight is that the model can no longer simply memorise the training set — it must learn the underlying structure of the data in order to 'fill in' the missing or noisy information. This produces more robust, generalisable features than a standard AE.

**Sparse Autoencoder.** A sparse autoencoder adds a penalty term that encourages most hidden unit activations to be close to zero for any given input. Only a few units should fire strongly for each example. Formally, we define $\hat{\rho}_j = \frac{1}{n}\sum_i h_j(\mathbf{x}^{(i)})$ as the average activation of hidden unit $j$, and add a KL divergence penalty that penalises $\hat{\rho}_j$ for deviating from a target sparsity $\rho$ (typically small, like 0.05). Sparse features tend to be more interpretable and generalisable than dense ones.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Denoising Autoencoder ─────────────────────────────────────────────
class DenoisingAE(nn.Module):
    def __init__(self, noise_std=0.2):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256,  64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear( 64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def corrupt(self, x):
        """Add Gaussian noise during training only."""
        if self.training:
            return x + self.noise_std * torch.randn_like(x)
        return x   # no corruption at inference

    def forward(self, x):
        x_noisy = self.corrupt(x).clamp(0, 1)   # add noise
        h       = self.encoder(x_noisy)          # encode noisy input
        x_hat   = self.decoder(h)                # decode to clean output
        return x_hat

# CRITICAL: Loss compares reconstruction with CLEAN input, not noisy input
dae = DenoisingAE(noise_std=0.3)
x_clean = torch.rand(16, 784)   # clean images
x_recon = dae(x_clean)          # internally corrupts, then reconstructs
loss    = nn.MSELoss()(x_recon, x_clean)  # compare with clean original
print(f'DAE reconstruction loss: {loss.item():.4f}')

# ── Sparse Autoencoder (sparsity via L1 penalty on activations) ───────
class SparseAE(nn.Module):
    def __init__(self, latent_dim=64, sparsity_weight=1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder = nn.Sequential(nn.Linear(784, latent_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 784), nn.Sigmoid())

    def forward(self, x):
        h     = self.encoder(x)
        x_hat = self.decoder(h)
        # L1 penalty on activations encourages sparsity
        sparsity_loss = self.sparsity_weight * h.abs().mean()
        return x_hat, sparsity_loss

sparse_ae = SparseAE(latent_dim=128)
x = torch.rand(16, 784)
x_hat, sp_loss = sparse_ae(x)
recon_loss = nn.MSELoss()(x_hat, x)
total_loss = recon_loss + sp_loss
print(f'Recon: {recon_loss:.4f}  Sparsity: {sp_loss:.4f}  Total: {total_loss:.4f}')
```

*Code 2 – Denoising and sparse autoencoders. The denoising AE adds noise during training but always computes the loss against the clean input — the model is forced to denoise. The sparse AE adds an L1 penalty on the activations, pushing most hidden units towards zero.*

---

## 3  The Latent Space Problem

With a trained autoencoder we can encode images into latent vectors and decode them back. It is tempting to use this for generation: sample a random latent vector and decode it. Unfortunately, standard autoencoders are poor generative models for a fundamental reason.

The encoder learns to map each training image to a specific point in latent space — the most efficient encoding for that particular image. Different classes cluster in different regions. Between the clusters, the latent space is uncharted territory: the decoder was never trained on latent vectors from those regions, so it produces meaningless or incoherent images when given a randomly sampled vector that falls in a gap.

Visualising a 2D latent space trained on MNIST makes this concrete: the digits 0–9 each form a distinct cluster with visible gaps between them. Interpolating through a gap produces nonsense. A useful generative model needs a latent space that is dense and continuous everywhere — any point in the space should decode to a plausible image.

> **The generation problem in one sentence.** A standard autoencoder learns to encode efficiently, not to produce a well-structured latent space. Efficiency and generativity are in tension: the most efficient encoding for reconstruction creates disjoint clusters that make generation by random sampling impossible.

---

## 4  Variational Autoencoders (VAEs)

### 4.1  The Key Idea: Encoding Distributions, Not Points

The VAE (Kingma & Welling, 2014) solves the latent space problem by changing what the encoder outputs. Instead of mapping each input to a single point in latent space, the encoder maps it to a probability distribution — specifically, a Gaussian with learned mean $\boldsymbol{\mu}$ and standard deviation $\boldsymbol{\sigma}$:

$$\text{Encoder:} \quad \mathbf{x} \to (\boldsymbol{\mu},\, \boldsymbol{\sigma}) \qquad \text{[two vectors of size latent\_dim]}$$

$$\text{Sample:} \quad \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \cdot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$\text{Decoder:} \quad \mathbf{z} \to \hat{\mathbf{x}}$$

Because $\mathbf{z}$ is sampled from a distribution centred at $\boldsymbol{\mu}$ rather than fixed at $\boldsymbol{\mu}$, the decoder is trained on a range of latent vectors for each training image, not just one. The neighbourhood around each encoding becomes decodable into something reasonable. This is local smoothness — nearby latent vectors decode to similar-looking images.

### 4.2  The Reparameterisation Trick

There is a subtle but critical problem: sampling is not differentiable. If we compute $\mathbf{z}$ by sampling from $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$, we cannot backpropagate gradients through the sampling step. The reparameterisation trick solves this by separating the randomness from the parameters:

$$\text{Instead of:} \quad \mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \qquad \text{[not differentiable w.r.t. } \boldsymbol{\mu}, \boldsymbol{\sigma}\text{]}$$

$$\text{Write as:} \quad \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \cdot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \qquad \text{[differentiable w.r.t. } \boldsymbol{\mu}, \boldsymbol{\sigma}\text{]}$$

The random variable $\boldsymbol{\varepsilon}$ is sampled from a fixed standard normal distribution — it carries all the randomness. The learnable parameters $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are deterministic functions of $\boldsymbol{\varepsilon}$, so gradients can flow back through $\mathbf{z}$ to the encoder. This elegant trick makes the entire VAE end-to-end differentiable.

### 4.3  The KL Divergence Loss Term

Local smoothness alone is not enough. Even with a stochastic encoder, the network could still learn to cluster different digit classes far apart in latent space (just as the standard AE does), as long as each cluster has a small variance. Sampling a random point from the gaps between clusters would still produce bad outputs.

The solution is to add a second term to the loss: the KL divergence between the encoded distribution $q(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ and a standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_\text{KL}\!\left[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\right] = \frac{1}{2} \sum_j \left(\mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1\right)$$

This term penalises the encoder for producing distributions that deviate from $\mathcal{N}(\mathbf{0}, \mathbf{I})$. It pushes all class clusters towards the centre of the latent space with similar spreads, making the entire latent space dense and navigable. The total VAE loss is the sum of reconstruction loss and KL loss, balanced by a weight $\beta$:

$$\mathcal{L}_\text{VAE} = \mathbb{E}\!\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right] + \beta \cdot D_\text{KL}\!\left[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\right]$$

The two terms are in tension: the reconstruction loss wants to create tight, distinct clusters for each class (easier to decode); the KL term wants to push all clusters onto the unit Gaussian. The network finds a compromise that satisfies both — clusters that are compact enough to decode well, but globally centred and overlapping enough to make the whole latent space usable.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════
# VARIATIONAL AUTOENCODER — side-by-side with standard AE
# The ONLY architectural difference is the encoder output and
# the reparameterisation trick. Everything else is identical.
# ════════════════════════════════════════════════════════════════════

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: outputs TWO vectors (mu and log_var) instead of one
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14×14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # 7×7
            nn.Flatten()                                           # 3136
        )
        self.fc_mu      = nn.Linear(3136, latent_dim)   # mean
        self.fc_log_var = nn.Linear(3136, latent_dim)   # log(σ²)

        # Decoder: identical to standard AE decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3136), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1,  3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h       = self.encoder_conv(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        """Reparameterisation trick: z = mu + std * eps, eps ~ N(0,1)."""
        std = torch.exp(0.5 * log_var)   # sigma = exp(log_var / 2)
        eps = torch.randn_like(std)       # sample from N(0,1)
        return mu + std * eps             # differentiable w.r.t. mu, std

    def decode(self, z):  return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)  # stochastic
        x_hat       = self.decode(z)
        return x_hat, mu, log_var

def vae_loss(x_hat, x, mu, log_var, beta=1.0):
    """
    VAE loss = Reconstruction loss + beta * KL divergence.
    Reconstruction: how well does x_hat match x?
    KL:             how close is N(mu, sigma^2) to N(0,1)?
    """
    # Reconstruction loss (MSE or BCE)
    recon = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)

    # KL divergence: (1/2) * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    # Using log_var = log(sigma^2) for numerical stability
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

    return recon + beta * kl, recon, kl

# ── Training step ─────────────────────────────────────────────────────
vae = VAE(latent_dim=20)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

x = torch.rand(32, 1, 28, 28)   # batch of images
opt.zero_grad()
x_hat, mu, log_var = vae(x)
loss, recon, kl = vae_loss(x_hat, x, mu, log_var, beta=1.0)
loss.backward()
opt.step()
print(f'Total: {loss:.4f}  Recon: {recon:.4f}  KL: {kl:.4f}')

# ── Generating new images (no encoder needed!) ────────────────────────
vae.eval()
with torch.no_grad():
    # Sample from the prior N(0,1) — the KL term ensures this is valid
    z_random = torch.randn(16, 20)   # 16 random latent vectors
    generated = vae.decode(z_random)
    print(f'Generated images shape: {generated.shape}')   # (16, 1, 28, 28)

# ── Latent space interpolation ────────────────────────────────────────
with torch.no_grad():
    mu1, _ = vae.encode(x[[0]])   # mean encoding of image 0
    mu2, _ = vae.encode(x[[1]])   # mean encoding of image 1
    # Interpolate in latent space (use means, not samples, for clarity)
    alphas   = torch.linspace(0, 1, 8)
    z_interp = torch.stack([a * mu2 + (1-a) * mu1 for a in alphas])
    x_interp = vae.decode(z_interp.squeeze(1))
    print(f'Interpolated sequence shape: {x_interp.shape}')  # (8, 1, 28, 28)
```

*Code 3 – The complete VAE. The only architectural difference from Code 1 (standard AE) is that `fc_mu` and `fc_log_var` replace the single `fc` layer, and the `reparameterise()` method provides the stochastic sampling. The `vae_loss()` function implements both the reconstruction term and the closed-form KL divergence for Gaussian latent variables.*

### 4.4  Generating New Images

Once trained, the VAE can generate new images without any input image at all. Because the KL term has enforced that the latent space follows $\mathcal{N}(\mathbf{0}, \mathbf{I})$, we can sample a random vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and feed it directly into the decoder. The decoder has been trained to handle exactly these inputs — it will produce a plausible new image.

This is the core value of the KL term: it converts the decoder into a usable image generator. The encoder is only needed during training to provide good starting points for $\mathbf{z}$. At inference time, generation is just: sample $\mathbf{z}$ from $\mathcal{N}(\mathbf{0}, \mathbf{I})$, compute $\text{decoder}(\mathbf{z})$.

### 4.5  Latent Space Arithmetic

One of the most striking properties of well-trained VAEs (and GANs) is that the latent space supports arithmetic. For face images: $\text{encode}(\text{woman with glasses}) - \text{encode}(\text{woman without glasses}) + \text{encode}(\text{man without glasses}) \approx$ latent vector that decodes to 'man with glasses'. This works because the KL regularisation forces different attributes to be represented as independently varying dimensions in the latent space, making addition and subtraction of attribute vectors meaningful.

---

## 5  Generative Adversarial Networks (GANs)

### 5.1  The Motivation: VAEs Are Blurry

VAEs produce well-structured latent spaces and generate coherent images, but the images are blurry. The reason is the MSE reconstruction loss: it penalises the average pixel error, which biases the network towards producing the mean of all plausible reconstructions rather than any single sharp one. When there is uncertainty about a fine detail (the exact texture of fur, the precise shape of a letter), the MSE-optimal prediction is a blurred average.

GANs (Goodfellow et al., 2014) take a completely different approach: instead of a pixel-level reconstruction loss, they use a learned discriminator network that judges whether an image looks real or fake. This provides a much richer training signal that pushes the generator towards producing sharp, photorealistic outputs — the discriminator can detect blurriness directly, whereas MSE cannot.

### 5.2  The Two-Network Architecture

A GAN consists of two networks trained simultaneously in a minimax game:

- **Generator** $G$: takes a random noise vector $\mathbf{z} \sim p(\mathbf{z})$ (typically $\mathcal{N}(\mathbf{0}, \mathbf{I})$) as input and produces a fake image $G(\mathbf{z})$. It never sees real images directly — it only receives feedback from the discriminator.
- **Discriminator** $D$: takes an image (real or fake) and outputs $D(\mathbf{x}) \in (0, 1)$, the estimated probability that $\mathbf{x}$ came from the real training distribution rather than the generator. It is a binary classifier.

### 5.3  The Loss Functions

The training objective is a minimax game. The discriminator wants to maximise its ability to tell real from fake; the generator wants to minimise the discriminator's ability:

$$\min_G \; \max_D \; \mathbb{E}_{\mathbf{x} \sim p_\text{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

This decomposes into two separate loss functions, one per network:

- **Discriminator loss**: maximise $\log D(\mathbf{x})$ for real images ($D$ should output $\approx 1$) plus $\log(1 - D(G(\mathbf{z})))$ for fake images ($D$ should output $\approx 0$). This is just binary cross-entropy on a balanced dataset of real and fake examples.
- **Generator loss**: minimise $\log(1 - D(G(\mathbf{z})))$, i.e. make the discriminator assign high probability to fake images. In practice, the alternative ('non-saturating') objective is used: maximise $\log D(G(\mathbf{z}))$, which provides stronger gradients when the discriminator is confident the image is fake.

> **Why is GAN training so hard?** $G$ and $D$ are trained simultaneously but with opposing objectives. If $D$ gets too strong, it assigns near-zero probability to every fake image — the generator's gradient vanishes and it stops learning (diminished gradient). If $G$ gets too strong, $D$ cannot distinguish real from fake and provides no useful signal. Mode collapse occurs when $G$ finds one or a few outputs that fool $D$, and stops producing variety. Balancing the two networks requires careful hyperparameter tuning and architectural choices.

```python
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════════
# DCGAN: Deep Convolutional GAN
# Generator: noise → image (uses ConvTranspose2d for upsampling)
# Discriminator: image → real/fake probability (uses strided Conv2d)
# ════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, features=64):
        super().__init__()
        # z_dim → 7×7×(features*2) → 14×14×features → 28×28×img_channels
        self.net = nn.Sequential(
            # Project noise to spatial feature map
            nn.ConvTranspose2d(z_dim, features*2, 7, 1, 0, bias=False),  # 7×7
            nn.BatchNorm2d(features*2), nn.ReLU(True),
            # Upsample to 14×14
            nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features), nn.ReLU(True),
            # Upsample to 28×28
            nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()    # output in [-1, 1]; normalise real images to [-1,1] too
        )

    def forward(self, z):
        # z shape: (batch, z_dim, 1, 1) — treat noise as a 1×1 'feature map'
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, features=64):
        super().__init__()
        # 28×28 → 14×14 → 7×7 → scalar
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features,   4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),    # Leaky ReLU — standard for D
            nn.Conv2d(features,     features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2,   1,          7, 1, 0, bias=False),
            nn.Sigmoid()   # D(x) = P(real)
        )

    def forward(self, x):  return self.net(x).view(-1)   # scalar per image

# ── Initialise weights with DCGAN paper convention ────────────────────
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

G = Generator(z_dim=100).apply(weights_init)
D = Discriminator().apply(weights_init)

# ── CRITICAL: two SEPARATE optimisers ─────────────────────────────────
# G and D are updated independently — never update both at once
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ── One training iteration ────────────────────────────────────────────
batch_size = 32
z_dim      = 100
real_images = torch.rand(batch_size, 1, 28, 28) * 2 - 1   # fake 'real' data

real_labels = torch.ones(batch_size)    # 1 = real
fake_labels = torch.zeros(batch_size)   # 0 = fake

# ── Step 1: Train Discriminator ───────────────────────────────────────
# 'Maximise log D(x) + log(1 - D(G(z)))'
opt_D.zero_grad()

# Loss on real images: D should output 1 for all of them
d_real = D(real_images)
loss_D_real = criterion(d_real, real_labels)

# Loss on fake images: D should output 0 for all of them
z = torch.randn(batch_size, z_dim, 1, 1)   # sample noise
fake_images = G(z).detach()                 # detach: don't update G here
d_fake = D(fake_images)
loss_D_fake = criterion(d_fake, fake_labels)

loss_D = loss_D_real + loss_D_fake
loss_D.backward()
opt_D.step()

# ── Step 2: Train Generator ───────────────────────────────────────────
# 'Minimise log(1 - D(G(z)))' ≡ 'Maximise log D(G(z))' (non-saturating)
opt_G.zero_grad()

z = torch.randn(batch_size, z_dim, 1, 1)
fake_images = G(z)          # generate again (this time we DO want gradients)
d_fake_for_G = D(fake_images)
# Use real_labels: G wants D to think fake images ARE real
loss_G = criterion(d_fake_for_G, real_labels)
loss_G.backward()
opt_G.step()

print(f'D loss: {loss_D.item():.4f}   G loss: {loss_G.item():.4f}')
# Healthy training: both losses should be around log(2) ≈ 0.69
# D >> G loss: D is dominating, G is getting no signal
# G >> D loss: G is dominating (or D is not training)

# ── Generating images at inference time ───────────────────────────────
G.eval()
with torch.no_grad():
    z = torch.randn(16, z_dim, 1, 1)
    generated = G(z)   # shape: (16, 1, 28, 28), values in [-1, 1]
    print(f'Generated: {generated.shape}')
```

*Code 4 – DCGAN training loop. The most important detail: two separate optimisers, updated in two completely separate steps. Step 1 updates $D$ only (detach the fake images so $G$ gets no gradient). Step 2 updates $G$ only (do not call `opt_D.step()`). Mixing these steps is the single most common GAN implementation bug.*

### 5.4  GAN Variants

**Conditional GAN (CGAN).** The original GAN has no control over what kind of image is generated. A conditional GAN (Mirza & Osindero, 2014) conditions both the generator and discriminator on a label $y$ (typically provided as a one-hot vector concatenated to the noise input). This allows controlled generation: 'generate a face with glasses', 'generate the digit 3'.

**DCGAN.** DCGAN (Radford et al., 2015) established a set of architectural best practices for using convolutions in GANs: strided convolutions instead of pooling in the discriminator, transposed convolutions in the generator, batch normalisation in both, LeakyReLU in the discriminator, and ReLU in the generator. These guidelines made GAN training significantly more stable and are still widely used.

**CycleGAN.** CycleGAN (Zhu et al., 2017) enables image-to-image translation without paired training data. Two generators ($G: X \to Y$ and $F: Y \to X$) are trained with a cycle consistency constraint: $F(G(\mathbf{x})) \approx \mathbf{x}$ and $G(F(\mathbf{y})) \approx \mathbf{y}$. This allows unpaired domain translation — turning horses into zebras without ever having a 'horse+matching zebra' training pair.

**Common Failure Modes:**

- **Mode collapse**: $G$ produces only one or a few outputs that fool $D$, ignoring most of the real distribution. Symptom: generated images all look very similar.
- **Diminished gradient**: $D$ becomes too good too quickly, assigning near-zero probability to all fake images. $G$'s gradient vanishes and it stops improving. Symptom: $G$ loss stops decreasing despite $D$ loss being low.
- **Non-convergence**: $G$ and $D$ oscillate rather than reaching equilibrium. Both losses fluctuate without clear trend. Symptom: erratic loss curves.

---

## 6  Side-by-Side Comparison: AE, VAE, and GAN

```python
import torch
import torch.nn as nn

# ── The three models share the same encoder/decoder backbone ──────────
# This makes the differences between them maximally clear.

def make_encoder(latent_dim):
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, latent_dim)
    )

def make_decoder(latent_dim):
    return nn.Sequential(
        nn.Linear(latent_dim, 256), nn.ReLU(),
        nn.Linear(256, 784), nn.Sigmoid()
    )

# ── STANDARD AE: encoder outputs one vector ───────────────────────────
class AE(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.enc = make_encoder(d)
        self.dec = make_decoder(d)
    def forward(self, x):
        return self.dec(self.enc(x.flatten(1)))
    # Generation: NOT possible without manual sampling heuristics

# ── VAE: encoder outputs (mu, log_var); reparameterisation trick ──────
class VAESimple(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.enc_shared = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.fc_mu      = nn.Linear(256, d)
        self.fc_lv      = nn.Linear(256, d)
        self.dec        = make_decoder(d)
    def forward(self, x):
        h   = self.enc_shared(x.flatten(1))
        mu  = self.fc_mu(h)
        lv  = self.fc_lv(h)
        std = torch.exp(0.5 * lv)
        z   = mu + std * torch.randn_like(std)   # reparameterisation
        return self.dec(z), mu, lv
    def generate(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.fc_mu.out_features)
            return self.dec(z).view(n, 1, 28, 28)

# ── GAN: no encoder at all at inference time ──────────────────────────
class SimpleGenerator(nn.Module):
    def __init__(self, z=100, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)
    def generate(self, n, z_dim=100):
        with torch.no_grad():
            z = torch.randn(n, z_dim)
            return self.net(z).view(n, 1, 28, 28)

# ── What sets them apart in a nutshell ───────────────────────────────
# AE:  train with MSE(x_hat, x)
#      latent space: unstructured — BAD for generation
#      generation:   not practical
#
# VAE: train with MSE(x_hat, x) + KL(N(mu,s²) || N(0,1))
#      latent space: structured, continuous — GOOD for generation
#      generation:   sample z~N(0,1), decode(z)  ← works well
#      weakness:     blurry outputs (MSE reconstruction loss)
#
# GAN: train G and D with adversarial losses (no reconstruction loss)
#      latent space: unstructured (no encoder needed at inference)
#      generation:   sample z~N(0,1), G(z)  ← very sharp outputs
#      weakness:     notoriously unstable training
```

*Code 5 – AE, VAE, and simple GAN generator side by side. The comment block at the bottom is the most important thing in this code: the one-sentence characterisation of what each model does, what its latent space looks like, how generation works, and what its main weakness is.*

---

## 7  Summary: The Generative Model Hierarchy

This lecture told a story of three models, each one motivated by the failure mode of the previous:

| Model | Latent space | Loss | Strengths | Weaknesses |
|---|---|---|---|---|
| AE | Unstructured clusters | MSE recon | Simple, fast, good features | Cannot generate — gaps in latent space |
| VAE | Smooth $\mathcal{N}(\mathbf{0},\mathbf{I})$ by KL | MSE + KL div | Structured generation, arithmetic | Blurry outputs (MSE averages) |
| GAN | No explicit structure | Adversarial BCE | Sharp, photorealistic images | Unstable training, mode collapse |
| Diffusion (Lec 13) | N/A | Denoising MSE | Best quality, stable training | Slow sampling (many steps) |

The progression from AE → VAE → GAN → Diffusion tracks the field's effort to produce better and better generative models. Each step introduced a new idea — regularising the latent space with KL divergence, replacing pixel-level loss with an adversarial discriminator, and finally (in Lecture 13) replacing the one-shot generation of GANs with a gradual iterative denoising process that is both more stable and higher quality. The reparameterisation trick (VAE) and the two-network adversarial game (GAN) are two of the most elegant ideas in deep learning and are worth studying in detail.

---

## References

- Kingma, D. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
- Goodfellow, I. et al. (2014). Generative Adversarial Networks. NeurIPS.
- Radford, A. et al. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN). ICLR 2016.
- Zhu, J.-Y. et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV.
- Mirza, M. & Osindero, S. (2014). Conditional Generative Adversarial Nets. arXiv.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Chapter 14 — Autoencoders. MIT Press.
- Introductory blog posts: jeremyjordan.me/autoencoders/ and jeremyjordan.me/variational-autoencoders/