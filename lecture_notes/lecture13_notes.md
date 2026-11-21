# Lecture 13
# Advanced Topics

*Deep Learning for Visual Recognition · Aarhus University*

These notes cover the final layer of modern deep learning: the information-theoretic foundations of cross-entropy loss, self-supervised learning and its pretext tasks, contrastive representation learning, diffusion models derived carefully from first principles, and CLIP as a bridge between vision and language.

---

## 1  Entropy, Cross-Entropy, and KL Divergence

The cross-entropy loss has been used throughout this course as if it were simply 'the loss function for classification'. This section explains where it actually comes from — and why it is the right choice — using Shannon's information theory.

### 1.1  Entropy: Measuring Uncertainty

Entropy $H(p)$ measures the average amount of information contained in one random sample drawn from a probability distribution $p$. Shannon's key insight: the information content of an event with probability $p_i$ is $-\log_2(p_i)$ bits — rare events carry more information than common ones. The average information over the whole distribution is:

$$H(p) = -\sum_i p_i \log_2(p_i)$$

The weather example from the slides makes this concrete: if tomorrow's weather is equally likely to be one of 8 states, hearing the forecast gives you exactly $\log_2(8) = 3$ bits of information. If one state (sunny) has 75% probability, the forecast carries less average information (0.81 bits) because you were already fairly sure it would be sunny.

High entropy means high uncertainty — the distribution is flat and every outcome is surprising. Zero entropy means the outcome is certain — one probability equals 1 and all others equal 0, and $-1 \cdot \log_2(1) = 0$ bits.

### 1.2  Cross-Entropy: Using the Wrong Code

Cross-entropy $H(p, q)$ arises when we encode a signal using a code optimised for distribution $q$, but the true distribution is actually $p$. It measures the average number of bits used, which is larger than necessary whenever $q \neq p$:

$$H(p, q) = -\sum_i p_i \log_2(q_i)$$

The key formula to notice: the log term uses $q$ (the predicted/encoded distribution) while the weight uses $p$ (the true distribution). If $q$ matches $p$ perfectly, $H(p,q) = H(p)$. Every mismatch wastes bits.

In neural network classification: $p$ is the true label distribution (a one-hot vector — probability 1 for the correct class, 0 for all others), and $q$ is the model's predicted distribution (the softmax output). Minimising the cross-entropy loss is therefore equivalent to finding the model whose predicted probabilities waste the fewest extra bits relative to the true distribution — an elegant information-theoretic justification for a loss function that might otherwise seem arbitrary.

### 1.3  KL Divergence: The Wasted Bits

The KL divergence $D_\text{KL}(p \| q)$ is the extra bits wasted by using $q$ instead of $p$ — the gap between cross-entropy and entropy:

$$D_\text{KL}(p \| q) = H(p, q) - H(p) = -\sum_i p_i \log_2\!\frac{q_i}{p_i}$$

KL divergence is always non-negative ($D_\text{KL} \geq 0$, with equality if and only if $p = q$). It is not symmetric — $D_\text{KL}(p \| q) \neq D_\text{KL}(q \| p)$ — so it is not a true distance metric, but a directed measure of how well $q$ approximates $p$.

We have already seen KL divergence in action in the VAE (Lecture 9), where the KL term in the loss pushed the encoder's latent distribution towards a standard normal prior. Minimising cross-entropy loss in classification is equivalent to minimising KL divergence between the predicted distribution and the one-hot target (since the entropy of a one-hot vector is fixed at zero).

---

## 2  Self-Supervised Learning (SSL)

### 2.1  The Labelling Bottleneck

ImageNet has 1.2 million labelled images, which took years of human effort to annotate. The internet has billions of unlabelled images. Medical imaging datasets may have millions of scans but very few expert annotations. The fundamental challenge of supervised deep learning is that the quality and quantity of annotations limits what can be learned.

Self-supervised learning (SSL) sidesteps this bottleneck by generating supervision from the data itself: create a pretext task whose labels can be derived automatically from raw data, train on that task, and then transfer the learned representations to downstream tasks where labels are scarce. The model never sees human-provided labels during pre-training.

### 2.2  Pretext Tasks

A pretext task is a self-supervised training objective whose labels are generated automatically from the raw data. The labels themselves are usually not what we care about — we care about the representations the model learns in order to solve the task. Common pretext tasks for images:

- **Rotation prediction**: Rotate images by 0°, 90°, 180°, 270° and train the model to predict which rotation was applied (4-way classification). The model must understand image semantics to distinguish, say, upside-down cats from right-way-up cats.
- **Relative patch location**: Extract two patches from the same image and train the model to predict the relative position of the second patch given the first (8-way classification for 8 possible neighbours). Requires recognising objects and their parts.
- **Jigsaw puzzle solving**: Scramble the patches of an image and train the model to predict the original permutation. Forces the model to reason about object structure.
- **Image inpainting**: Mask out a region of the image and train an encoder-decoder to reconstruct the missing content from the surroundings. Requires understanding scene context.
- **Colourisation**: Convert images to grayscale and train the model to predict the original colours. Forces learning of object-level semantic representations (grass is green; sky is blue).

The general principle: harder pretext tasks require deeper semantic understanding and therefore produce better features. The representations are evaluated not by performance on the pretext task itself, but on downstream tasks — typically by freezing the learned encoder and training only a linear classifier on top.

### 2.3  Masked Autoencoders (MAE)

MAE (He et al., 2021) applies the BERT-style masked prediction pretext task to Vision Transformers. A large fraction (typically 75%) of image patches are randomly masked, and the model is trained to reconstruct the pixel values of the masked patches from the remaining visible ones. The high masking ratio is crucial: images have strong spatial redundancy, so a low masking ratio can be solved by simple texture interpolation. Masking 75% forces the model to develop genuine semantic understanding of object structure in order to fill in large missing regions.

MAE uses an asymmetric encoder-decoder architecture for efficiency: only the ~25% visible tokens are processed by the large encoder (no mask tokens in the encoder), and a lightweight decoder reconstructs the full image including the masked patches. This means the heavy encoder processes only 1/4 of the tokens, making training very fast.

---

## 3  Contrastive Representation Learning

### 3.1  The Core Idea: Learning by Comparing

Contrastive learning trains an encoder to produce embeddings where similar inputs are close and dissimilar inputs are far apart in the embedding space. Unlike pretext tasks that predict specific transformations, contrastive learning defines similarity directly — no task-specific knowledge is baked in. The approach generalises naturally across domains.

The challenge is preventing collapse — the trivial solution where the encoder ignores all inputs and maps everything to the same point, which achieves zero contrastive loss without learning anything useful. Different methods have invented different collapse prevention mechanisms.

### 3.2  Triplet Loss

The simplest contrastive objective works with triplets: an anchor $\mathbf{x}$, a positive example $\mathbf{x}^+$ (same class/object as $\mathbf{x}$), and a negative example $\mathbf{x}^-$ (different class/object). The loss pushes the anchor closer to the positive than to the negative, with a margin $\varepsilon$:

$$\mathcal{L}_\text{triplet} = \max\!\left(0,\; d(\mathbf{x}, \mathbf{x}^+) - d(\mathbf{x}, \mathbf{x}^-) + \varepsilon\right)$$

The margin $\varepsilon$ prevents the trivial solution of collapsing all distances to zero. The $\max(0, \ldots)$ clips the loss to zero whenever the negative is already far enough away — no gradient flows when the triplet is already correctly ordered with enough margin.

### 3.3  SimCLR

SimCLR (Chen et al., 2020) generalises from triplets to batches and uses augmentation to create positive pairs automatically. Given a batch of $N$ images, two augmented views of each image are created, giving $2N$ images total. For each image, its augmented twin is the positive; all other $2(N-1)$ images are negatives. The NT-Xent loss (Normalised Temperature-scaled Cross-Entropy) is:

$$\mathcal{L} = -\log \frac{\exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\right)}{\sum_{k \neq i} \exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\right)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T\mathbf{v} / (\|\mathbf{u}\| \cdot \|\mathbf{v}\|)$ is cosine similarity and $\tau$ is a temperature hyperparameter. Notice this is just cross-entropy on a $(2N-1)$-way classification problem: 'find the positive pair among all $2N-1$ other images in the batch.' This connection to cross-entropy (Section 1) is elegant — the contrastive loss is the classification loss for the task of identifying augmented pairs.

SimCLR's key findings: (1) data augmentation composition is crucial — colour distortion combined with random cropping is far more effective than either alone; (2) a nonlinear projection head MLP between the encoder and the loss improves quality of representations; (3) large batch sizes (4096–8192) provide more negatives and substantially improve performance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── Contrastive losses ────────────────────────────────────────────────

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet loss: push anchor closer to positive than to negative.
    All inputs: (B, d) normalised embedding vectors.
    """
    # Use cosine distance (1 - cosine similarity)
    d_pos = 1 - F.cosine_similarity(anchor, positive)
    d_neg = 1 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()

def nt_xent_loss(z1, z2, temperature=0.07):
    """
    NT-Xent loss (SimCLR). Given two augmented views of a batch:
        z1, z2: (B, d) L2-normalised embeddings
    For each image, its twin in the other view is the positive;
    all other 2(B-1) images are negatives.
    """
    B = z1.size(0)
    # Stack both views: (2B, d)
    z = torch.cat([z1, z2], dim=0)
    # L2 normalise
    z = F.normalize(z, dim=1)
    # Pairwise cosine similarity: (2B, 2B)
    sim = torch.mm(z, z.t()) / temperature
    # Remove self-similarity from denominator
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float('-inf'))
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

# ── SimCLR data augmentation pipeline ────────────────────────────────
class SimCLRAugment:
    """Apply two random augmentations to each image to create a positive pair."""
    def __init__(self, img_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # colour distortion
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        # Return two independently augmented views of the same image
        return self.transform(x), self.transform(x)

# ── SimCLR model (encoder + projection head) ─────────────────────────
class SimCLR(nn.Module):
    def __init__(self, encoder, d_encoder, d_proj=128):
        super().__init__()
        self.encoder = encoder
        # Projection head: 2-layer MLP with ReLU — crucial for SimCLR quality
        self.projector = nn.Sequential(
            nn.Linear(d_encoder, d_encoder), nn.ReLU(),
            nn.Linear(d_encoder, d_proj)
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)  # representations (used for downstream tasks)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)  # projections (used for contrastive loss only)
        z2 = self.projector(h2)
        return h1, h2, z1, z2

# ── Training step ─────────────────────────────────────────────────────
import torchvision.models as models
backbone  = models.resnet18(weights=None)
backbone.fc = nn.Identity()   # remove classification head
model = SimCLR(backbone, d_encoder=512, d_proj=128)
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

# Fake batch: 32 unlabelled images — NO labels needed!
x1 = torch.randn(32, 3, 32, 32)   # view 1 of each image
x2 = torch.randn(32, 3, 32, 32)   # view 2 of each image

opt.zero_grad()
_, _, z1, z2 = model(x1, x2)
loss = nt_xent_loss(z1, z2, temperature=0.07)
loss.backward()
opt.step()
print(f'SimCLR loss: {loss.item():.4f}')  # should be near log(2*B-1) ≈ 4.1 initially
```

*Code 1 – Triplet loss and NT-Xent (SimCLR) from scratch. The `nt_xent_loss` is exactly `F.cross_entropy` on a $(2B)$-way problem — confirming the Section 1 intuition that contrastive learning is classification in disguise. Note that the model is trained entirely without labels — the positive pairs come from augmentation alone.*

---

## 4  Diffusion Models

### 4.1  The Motivation: Iterative Refinement

GANs generate images in a single forward pass through the generator — fast, but notoriously hard to train (mode collapse, vanishing gradients, training instability). VAEs generate in a single decoder forward pass — stable to train, but blurry. Diffusion models take a fundamentally different approach: spread the generation task over many small, easy denoising steps. Each step is individually simple, making training stable, but the cumulative effect produces the highest-quality images of any generative model to date.

The key insight from denoising autoencoders (Lecture 9) is that neural networks can learn to undo noise — they just need to be shown corrupted-and-clean pairs. Diffusion models scale this idea: define a precise noise schedule over $T$ steps ($T = 1{,}000$ is typical), train a single network to reverse one step of the schedule, and at inference time repeat that network $T$ times, each time removing a little noise.

### 4.2  The Forward Diffusion Process

The forward process is a fixed (non-learned) Markov chain that gradually adds Gaussian noise to a clean image $\mathbf{x}_0$ over $T$ steps. At each step $t$, a small amount of noise is added according to a noise schedule $\beta_1 < \beta_2 < \cdots < \beta_T < 1$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t} \cdot \mathbf{x}_{t-1},\; \beta_t \mathbf{I}\right)$$

The mean at each step is the previous image scaled down by $\sqrt{1-\beta_t}$ — this factor is slightly less than 1, so the image gradually fades. The variance is $\beta_t$, which increases over time. After $T$ steps with a well-chosen schedule, $\mathbf{x}_T$ is indistinguishable from pure Gaussian noise. This is the signal destruction phase — the forward process needs no learning.

**Computing $\mathbf{x}_t$ directly from $\mathbf{x}_0$.** A critical mathematical property: because each step adds independent Gaussian noise, we can compute $\mathbf{x}_t$ at any arbitrary time step $t$ directly from $\mathbf{x}_0$ in a single operation, without simulating all intermediate steps. Letting $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t} \cdot \mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I}\right)$$

$$\text{So:} \qquad \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \cdot \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \cdot \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This reparameterisation is essential for efficient training: we can sample any time step $t$ uniformly at random, immediately compute the noisy image $\mathbf{x}_t$ without simulation, and train the network on that example. Without this shortcut, training would require simulating the full $T$-step chain for every training image.

### 4.3  The Reverse Diffusion Process

The reverse process learns to undo the forward process one step at a time. It is parameterised as a Gaussian:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I}\right)$$

The learned mean $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ is what the neural network must predict — the best estimate of what the image looked like one step earlier, given the current noisy version. It turns out that rather than predicting the mean directly, it is more effective to predict the noise $\boldsymbol{\varepsilon}$ that was added — then the mean can be computed analytically. The network output $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$ is a noise prediction, and the training loss is simply MSE between predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol{\varepsilon}}\!\left[\left\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\cdot\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\cdot\boldsymbol{\varepsilon},\; t\right)\right\|^2\right]$$

This is a denoising objective — the network sees a noisy image and predicts the noise. The time step $t$ is injected as an additional input (typically via sinusoidal embeddings added to intermediate feature maps in the U-Net), telling the network how much noise to expect.

### 4.4  Training and Sampling Algorithms

The training and sampling algorithms are elegantly simple once the mathematics above is established:

**Training (Algorithm 1)**

- Sample a clean image $\mathbf{x}_0$ from the training data.
- Sample a random time step $t$ uniformly from $\{1, \ldots, T\}$.
- Sample noise $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
- Compute the noisy image: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \cdot \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \cdot \boldsymbol{\varepsilon}$.
- Predict the noise: $\hat{\boldsymbol{\varepsilon}} = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$.
- Take a gradient step on $\|\boldsymbol{\varepsilon} - \hat{\boldsymbol{\varepsilon}}\|^2$ and repeat.

**Sampling (Algorithm 2 — generating new images)**

- Start from pure noise: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
- For $t = T, T-1, \ldots, 1$: predict $\hat{\boldsymbol{\varepsilon}} = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$, compute the denoised estimate $\hat{\mathbf{x}}_0$, then sample $\mathbf{x}_{t-1}$ using the reverse Gaussian formula.
- Return $\mathbf{x}_0$ — the generated image.

### 4.5  The U-Net Denoiser

The denoising network $\boldsymbol{\varepsilon}_\theta$ is a U-Net — an encoder-decoder with skip connections that preserves spatial detail. It takes as input the noisy image $\mathbf{x}_t$ (same shape as a regular image) and outputs the predicted noise (same shape). The time step $t$ is converted to a sinusoidal embedding and added to every residual block of the U-Net, allowing the network to behave differently at each noise level. Without this temporal conditioning, the network would be asked to denoise an image without knowing how noisy it is — like being asked to correct a text without knowing how many errors to expect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ════════════════════════════════════════════════════════════════════
# DDPM (Denoising Diffusion Probabilistic Model) — key components
# ════════════════════════════════════════════════════════════════════

class NoiseSchedule:
    """
    Pre-computes the noise schedule constants used in DDPM.
    Linear schedule from beta_start to beta_end over T steps.
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T    = T
        betas     = torch.linspace(beta_start, beta_end, T)     # β₁ ... βₜ
        alphas    = 1.0 - betas                                  # αₜ = 1 - βₜ
        alpha_bar = torch.cumprod(alphas, dim=0)                 # ᾱₜ = Πₛαₛ

        # Store all quantities needed for forward process and loss
        self.betas         = betas
        self.alphas        = alphas
        self.alpha_bar     = alpha_bar
        self.sqrt_ab       = torch.sqrt(alpha_bar)               # √ᾱₜ
        self.sqrt_one_m_ab = torch.sqrt(1 - alpha_bar)           # √(1-ᾱₜ)

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: compute xₜ from x₀ in a single step.
        xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab_t     = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sqrt_one_m_t  = self.sqrt_one_m_ab[t].view(-1, 1, 1, 1)
        return sqrt_ab_t * x0 + sqrt_one_m_t * noise, noise

# ── Sinusoidal time embedding (tells the U-Net what t is) ────────────
class SinusoidalTimeEmbed(nn.Module):
    """Convert scalar time step t into a d-dimensional embedding vector."""
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        half  = self.d // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, d)

# ── Minimal U-Net block with time conditioning ────────────────────────
class ResBlock(nn.Module):
    """Residual block that accepts a time embedding."""
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, channels)
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.norm2  = nn.GroupNorm(8, channels)
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding to every spatial position
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h   # residual connection

# ── DDPM training step ────────────────────────────────────────────────
def ddpm_train_step(model, schedule, x0, optimizer):
    """
    One training step of DDPM.
    model:    U-Net that predicts noise given (noisy_image, time_step)
    schedule: NoiseSchedule
    x0:       (B, C, H, W) clean images from training set
    """
    B = x0.size(0)

    # 1. Sample random time steps
    t = torch.randint(0, schedule.T, (B,), device=x0.device)

    # 2. Sample noise and compute noisy images (single-step forward process)
    noise = torch.randn_like(x0)
    xt, noise = schedule.q_sample(x0, t, noise)

    # 3. Predict the noise (the network's job)
    noise_pred = model(xt, t)   # U-Net takes noisy image + time step

    # 4. Simple MSE loss between true and predicted noise
    loss = F.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ── DDPM sampling (inference — generates new images) ──────────────────
@torch.no_grad()
def ddpm_sample(model, schedule, shape, device):
    """
    Generate images by reversing the diffusion process.
    shape: e.g. (4, 3, 64, 64) — 4 images, RGB, 64×64
    """
    model.eval()
    # Start from pure noise
    x = torch.randn(shape, device=device)

    for t_val in reversed(range(schedule.T)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x, t)

        # Compute the denoised estimate and reverse-process sample
        alpha     = schedule.alphas[t_val]
        alpha_bar = schedule.alpha_bar[t_val]
        beta      = schedule.betas[t_val]

        # Estimate x₀ from xₜ and predicted noise
        x0_pred = (x - (1-alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Sample xₜ₋₁  (add a little noise unless t=0)
        mean  = (x - beta / (1-alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
        if t_val > 0:
            noise = torch.randn_like(x)
            x = mean + beta.sqrt() * noise
        else:
            x = mean   # final step: no noise added

    return x.clamp(-1, 1)

# ── Quick shape check ─────────────────────────────────────────────────
schedule = NoiseSchedule(T=1000)
x0 = torch.randn(4, 3, 32, 32)   # 4 clean 32×32 RGB images
t  = torch.randint(0, 1000, (4,))
xt, eps = schedule.q_sample(x0, t)
print(f'Clean images:  {x0.shape}')   # (4, 3, 32, 32)
print(f'Noisy images:  {xt.shape}')   # (4, 3, 32, 32) — same shape
print(f'True noise:    {eps.shape}')  # (4, 3, 32, 32) — what the U-Net predicts

# Verify the noise schedule: at t=999, image should be mostly noise
t_late = torch.full((4,), 999)
xt_late, _ = schedule.q_sample(x0, t_late)
print(f'Signal/noise ratio at t=0:   {x0.std():.3f} / 0.000')
print(f'Signal/noise ratio at t=999: {(schedule.sqrt_ab[999]*x0.std()):.3f} / {schedule.sqrt_one_m_ab[999]:.3f}')
```

*Code 2 – Complete DDPM implementation: noise schedule, forward process (`q_sample`), time embedding, residual block with time conditioning, training step, and sampling loop. The training step is just 4 lines of mathematics: sample $t$, compute noisy image, predict noise, compute MSE. The sampling loop runs those 4 lines in reverse $T=1{,}000$ times.*

---

## 5  Conditional Diffusion Models

### 5.1  Adding Conditioning Signals

Unconditional diffusion models generate images from the training distribution at random. Conditional diffusion models steer generation towards specific content by adding an extra conditioning signal $y$ to the denoiser: the network now predicts $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, y)$ instead of $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$. The conditioning can be a class label (generate a dog), a text embedding (generate a red rose in the style of Van Gogh), or an input image (image-to-image translation).

### 5.2  Classifier Guidance

One way to achieve strong conditioning without training a new conditional model is to mix the unconditional diffusion model's noise prediction with the gradient of a separately trained classifier $p(y \mid \mathbf{x}_t)$:

$$\tilde{\boldsymbol{\varepsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p(y \mid \mathbf{x}_t)$$

The classifier gradient $\nabla_{\mathbf{x}_t} \log p(y \mid \mathbf{x}_t)$ points in the direction that makes $\mathbf{x}_t$ look more like class $y$. Scaling this by the guidance strength $s$ allows trading off diversity for fidelity: high $s$ produces images that strongly match the class but with less variety; $s=0$ produces the unconditional distribution. Note that the classifier must be trained on noisy images at various time steps — a standard classifier trained on clean images will not provide useful gradients for highly noised $\mathbf{x}_t$.

### 5.3  Classifier-Free Guidance (CFG)

Classifier-free guidance (Ho & Salimans, 2022) is now the dominant approach. Instead of a separate classifier, the diffusion model itself is trained to be both conditional and unconditional simultaneously: during training, the conditioning signal $y$ is randomly dropped (replaced with a null token) with some probability. At inference, both conditional and unconditional predictions are computed and combined:

$$\tilde{\boldsymbol{\varepsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot \left(\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, \emptyset)\right)$$

This requires only one model — no separate classifier — and produces excellent results. The guidance scale $s$ controls the trade-off: $s=1$ gives the pure conditional prediction; $s>1$ amplifies the conditioning by extrapolating away from the unconditional prediction. Stable Diffusion, DALL-E 2, and Imagen all use CFG.

---

## 6  CLIP: Contrastive Language-Image Pre-training

### 6.1  Bridging Vision and Language

CLIP (Radford et al., OpenAI 2021) trains an image encoder and a text encoder jointly using contrastive learning on 400 million image-text pairs scraped from the internet. The training objective is simple: for each batch of $N$ image-text pairs, the image and text embeddings of matching pairs should be close in embedding space, and non-matching pairs should be far apart — exactly the NT-Xent objective from Section 3.3 applied across modalities.

$$\mathcal{L}_\text{CLIP} = \text{NT-Xent}\!\left(\{\text{image\_embeds}\},\, \{\text{text\_embeds}\}\right)$$

### 6.2  Zero-Shot Image Classification

CLIP's most remarkable capability is zero-shot classification — classifying images into categories it has never been explicitly trained on. The procedure:

- For each class, create a text prompt: 'a photo of a [class].'
- Embed all class prompts with the text encoder — this gives one text embedding per class.
- Embed the test image with the image encoder.
- Classify by finding the class whose text embedding has the highest cosine similarity to the image embedding.

No task-specific fine-tuning is needed — the shared embedding space means that an image of a labrador retriever and the text 'a photo of a labrador retriever' end up close to each other by virtue of the contrastive training. CLIP achieves competitive accuracy on ImageNet with no training on ImageNet labels at all.

### 6.3  CLIP as a Foundation for Generation

CLIP's shared image-text embedding space enables text-guided image generation. The key insight (connecting back to the 'inverting ConvNets' idea from Lecture 10): to generate an image matching a text prompt, find an image whose embedding is close to the text embedding. In diffusion models, this is implemented through CLIP-guided diffusion — the CLIP similarity score is used as the conditioning signal, guiding the reverse diffusion process towards images that match the text prompt. Stable Diffusion and DALL-E 2 are both built on this principle.

---

## 7  Latent Diffusion Models and Stable Diffusion

Pixel-space diffusion models (Section 4) apply the forward and reverse processes directly to image pixels. For high-resolution images, this is expensive: a $512 \times 512$ RGB image has 786,432 pixel values, and the U-Net must process all of them at every one of the $T=1{,}000$ denoising steps. Latent Diffusion Models (LDMs, Rombach et al. 2022) — the foundation of Stable Diffusion — solve this by moving diffusion into a compressed latent space:

- **Step 1 — Train a VAE**: A convolutional VAE encoder compresses images from pixel space ($512 \times 512 \times 3$) to a much smaller latent space (e.g. $64 \times 64 \times 4$) — a compression factor of $48\times$. The decoder reconstructs the original resolution.
- **Step 2 — Apply diffusion in latent space**: The forward and reverse diffusion processes operate on the $64 \times 64 \times 4$ latent codes, not the full-resolution pixels. The U-Net denoiser is much smaller and faster.
- **Step 3 — Condition on text via cross-attention**: Text prompts are encoded with a text transformer (CLIP's text encoder). The U-Net's intermediate layers have cross-attention layers where the latent image representation attends to the text embedding at each denoising step.

The result is dramatically more efficient: Stable Diffusion generates a $512 \times 512$ image in seconds rather than minutes, requires far less GPU memory, and can be run on consumer hardware. The quality is comparable to or better than pixel-space diffusion because the VAE's encoder/decoder learns a semantically meaningful latent space where the diffusion process is easier to learn.

---

## 8  Summary: The Modern Deep Learning Landscape

This lecture covered the frontier of modern deep learning. The connecting thread running through every topic is the question of how to learn useful representations from data — with less supervision, from more modalities, and at higher quality:

| Topic | Core idea | Key innovation | PyTorch entry point |
|---|---|---|---|
| Cross-entropy | Measure mismatch between predicted and true distributions | Information-theoretic justification for classification loss | `nn.CrossEntropyLoss()` |
| SSL pretext tasks | Generate labels from raw data | No human annotation needed; diverse pre-text tasks | Custom transforms + standard model |
| MAE | Reconstruct 75% masked patches | Efficient: encoder sees only 25% of tokens | ViT encoder + lightweight decoder |
| Contrastive (SimCLR) | Pull similar pairs together, push different apart | NT-Xent = $N$-way cross-entropy; augmentation creates pairs | `F.cosine_similarity` + `F.cross_entropy` |
| Diffusion (DDPM) | Iterative denoising over $T$ steps | Noise prediction; single-step forward; stable training | U-Net + noise schedule |
| CFG | Guide diffusion with text/class signal | No separate classifier; one model does both | Dual forward pass (conditional + unconditional) |
| CLIP | Joint image-text embedding via contrastive learning | Zero-shot classification; shared vision-language space | `transformers CLIPModel` |
| Latent Diffusion | Diffusion in VAE latent space, not pixels | $48\times$ compression → fast, memory-efficient generation | Stable Diffusion via `diffusers` |

The lecture closes the loop on the entire course. We began with supervised learning (Lecture 2), built up CNNs (Lectures 4–8), learned about generative models (Lecture 9), and saw how Transformers changed the landscape (Lectures 11–12). Lecture 13 synthesises these threads: self-supervised learning removes the labelling bottleneck, contrastive learning provides a general framework for representation learning without labels, diffusion models achieve generation quality that GANs and VAEs could not match, and CLIP unifies vision and language in a single shared space. Together, these techniques underpin essentially every state-of-the-art system in computer vision today — from Stable Diffusion to GPT-4V to autonomous driving models.

---

## 9  Exercises
The exercises are representative of the types of questions you might encounter at the written exam (except that the exam questions will be formulated as multiple choice). You are welcome to work on the exercises during TØ on Thursdays.

### Exercise 1 — Entropy

**(a)** What does the entropy $H(p)$ of a distribution measure? Why does a rare event carry more information than a common one?

**(b)** A weather station reports tomorrow's weather as one of four states (sunny, cloudy, rain, snow). If all four states are equally likely, how many bits of information does the report carry on average? How many bits if the state is certain to be sunny? (Hint: $\log_2 4 = 2$.)

**(c)** On another day the four states have probabilities $\tfrac12, \tfrac14, \tfrac18, \tfrac18$. Compute the entropy. Is it higher or lower than in (b), and why does this make sense?

---

### Exercise 2 — Cross-Entropy, KL Divergence and Classification

$$H(p, q) = -\sum_i p_i \log_2 q_i \qquad D_\text{KL}(p \,\|\, q) = H(p, q) - H(p)$$

**(a)** Explain what cross-entropy $H(p, q)$ measures using the idea of an encoding. Which distribution do we assume when we design the code, and which is the true distribution?

**(b)** Which of the two distributions appears inside the logarithm and which gives the weights? What is the relation between $H(p,q)$ and $H(p)$, and when are they equal?

**(c)** What is the KL divergence, and what does it measure in words?

**(d)** Name the loss used in logistic regression (binary classification) and in softmax regression (multi-class). How are both related to cross-entropy?

**(e)** PyTorch's `nn.CrossEntropyLoss` uses the natural logarithm instead of $\log_2$. How does this change the loss values, and does it change which model minimises the loss?

---

### Exercise 3 — Self-Supervised Learning and Pretext Tasks

**(a)** Explain the labelling bottleneck and how self-supervised learning (SSL) sidesteps it.

**(b)** Compare SSL with supervised learning and with unsupervised learning. Where do the labels come from in each, and what do we want to obtain from the SSL model in the end?

**(c)** What is a pretext task? Why do we usually not care about performance on the pretext task itself?

**(d)** For each of the pretext tasks below, state what the input is and what the automatically generated label is:

- Rotation prediction
- Relative patch location
- Jigsaw puzzle solving
- Colourisation

**(e)** A hospital has five million unlabelled chest X-rays and only a few hundred labelled ones. Describe how you would use SSL in this situation. What do you do after the pretext task, and how is the learned representation typically evaluated?

---

### Exercise 4 — Masked Autoencoders

**(a)** Describe how a Masked Autoencoder (MAE) is trained: what is the input, what is masked, and what is the model asked to predict?

**(b)** Which type of network is used as the encoder in MAE (from Lecture 12)?

---

### Exercise 5 — Contrastive Learning and the Triplet Loss

$$\mathcal{L}_\text{triplet} = \max\!\left(0,\; d(\mathbf{x}, \mathbf{x}^+) - d(\mathbf{x}, \mathbf{x}^-) + \varepsilon\right)$$

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    d_pos = 1 - F.cosine_similarity(anchor, positive)
    d_neg = 1 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()
```

**(a)** What is the basic idea of contrastive representation learning? How does it differ from a pretext task such as rotation prediction?

**(b)** What are the anchor, the positive and the negative in the triplet loss? Which one is pulled towards the anchor and which one is pushed away?

**(c)** What is the role of the margin $\varepsilon$? What would happen if the margin were zero and the network could simply map every input to the same point? This failure is called what?

**(d)** Draw (or describe) the network architecture used for training with triplets: how many encoders are used, and what is the relation between their weights?

**(e)** For supervised data, how would you choose positives and negatives? For unlabelled data, how could you create a positive for an image?

---

### Exercise 6 — Forward Diffusion

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I}\right) \qquad \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon$$

**(a)** What is the purpose of diffusion models, and what is the role of the forward process in them? Does the forward process contain any learned parameters?

**(b)** Explain the role of $t$, $\mathbf{x}_0$, $\mathbf{x}_T$ and the noise scheduler $\beta_t$. Why does $\beta_t$ increase over time?

**(c)** In each step, the previous image is multiplied by $\sqrt{1-\beta_t}$, which is slightly less than 1. What is the effect of this factor, and what does it do together with the added noise to the overall variance?

**(d)** What does the second formula allow us to do? Explain in your own words why it is possible to jump directly from $\mathbf{x}_0$ to $\mathbf{x}_t$.

**(e)** Why is it important for training that $\mathbf{x}_t$ can be computed from $\mathbf{x}_0$ in a single step? What would training look like without this property?

---

### Exercise 7 — Reverse Diffusion and Training

```python
def ddpm_train_step(model, schedule, x0, optimizer):
    B = x0.size(0)
    t = torch.randint(0, schedule.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt, noise = schedule.q_sample(x0, t, noise)
    noise_pred = model(xt, t)
    loss = F.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**(a)** The reverse process is $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$. What is the network asked to do?

**(b)** In practice the network does not predict the mean $\boldsymbol{\mu}_\theta$ directly. What does it predict instead, and what is the input and output shape of the network compared with a regular image?

**(c)** Go through the code line by line and match each line to a step of the training algorithm from the lecture.

**(d)** What is the loss measuring, and why can this be called a denoising objective? Which model from Lecture 9 is it related to?

---

### Exercise 8 — Sampling, the U-Net and Image Quality

```python
x = torch.randn(shape)                                # start
for t_val in reversed(range(schedule.T)):
    noise_pred = model(x, t)
    mean = (x - beta / (1-alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
    if t_val > 0:
        x = mean + beta.sqrt() * torch.randn_like(x)
    else:
        x = mean
return x
```

**(a)** Describe the sampling algorithm: what does it start from, what happens in each iteration, and what is returned?

**(b)** How many times is the network evaluated to generate one image if $T = 1000$? Compare this with the number of forward passes needed by a GAN generator or a VAE decoder, and explain what is the main practical disadvantage of diffusion models.

**(c)** How is the time step $t$ given to the U-Net? Which technique from Lecture 12 (Vision Transformers) is used to turn the integer $t$ into a vector? Explain why this "temporal component" is needed.

**(d)** Give two reasons why diffusion models tend to generate higher-quality images than GANs and VAEs.

---

### Exercise 9 — Conditional Diffusion and Guidance

**(a)** What is a conditional diffusion model? Give three examples of conditioning signals $y$.

**(b)** How does the denoising network change compared with an unconditional diffusion model?

---

### Exercise 11 — CLIP and Zero-Shot Classification

**(a)** What are the two networks of CLIP, and what kind of data is it trained on?

**(b)** For a batch of $N$ image–text pairs, a similarity matrix of size $N \times N$ is computed. Which entries are positives and which are negatives?

**(c)** Describe step by step how CLIP classifies an image into one of $K$ classes without any training on those classes (zero-shot). What do you compare with what?

**(d)** Why does the text prompt look like "a photo of a [class]" instead of just the class name?

**(e)** What is the advantage of zero-shot classification compared with a supervised classifier when a new class is added?

**(f)** Explain how CLIP's shared embedding space can be used to guide image generation from a text prompt, and relate this to the idea from Lecture 10 of finding an image that matches a target representation.

---

## References

- Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models (DDPM). NeurIPS.
- He, K. et al. (2021). Masked Autoencoders Are Scalable Vision Learners (MAE). CVPR 2022.
- Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.
- Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). ICCV.
- Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS Workshop.
- Annotated diffusion (HuggingFace blog): huggingface.co/blog/annotated-diffusion
- Lilian Weng's diffusion overview: lilianweng.github.io/posts/2021-07-11-diffusion-models/