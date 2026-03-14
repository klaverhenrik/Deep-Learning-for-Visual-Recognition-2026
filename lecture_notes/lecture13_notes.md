# Lecture 13 — Advanced Topics

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes cover the final layer of modern deep learning: the information-theoretic foundations of cross-entropy loss, self-supervised learning, contrastive representation learning, diffusion models derived carefully from first principles, and CLIP as a bridge between vision and language.

---

## 1  Entropy, Cross-Entropy, and KL Divergence

### 1.1  Entropy

Shannon entropy $H(\mathbf{p})$ measures the average information content of one sample from a probability distribution:

$$H(\mathbf{p}) = -\sum_i p_i \log_2 p_i$$

High entropy = high uncertainty (flat distribution). Zero entropy = certainty (one probability equals 1).

**Worked example**: 75% sunny / 25% rainy weather.
- If forecast says rainy: information = $-\log_2 0.25 = 2$ bits (rare event, high information)
- If forecast says sunny: information = $-\log_2 0.75 = 0.415$ bits (common event, low information)
- Average (entropy): $0.75 \times 0.415 + 0.25 \times 2 = 0.81$ bits

### 1.2  Cross-Entropy

Cross-entropy $H(\mathbf{p}, \mathbf{q})$ measures the average bits used when coding a distribution $\mathbf{p}$ with a code optimised for $\mathbf{q}$:

$$H(\mathbf{p}, \mathbf{q}) = -\sum_i p_i \log_2 q_i$$

In neural network classification: $\mathbf{p}$ is the true one-hot label distribution; $\mathbf{q}$ is the model's softmax output. Minimising cross-entropy = finding the model whose predictions waste the fewest extra bits relative to the true distribution.

### 1.3  KL Divergence

The extra bits wasted by using $\mathbf{q}$ instead of $\mathbf{p}$:

$$D_\text{KL}(\mathbf{p} \| \mathbf{q}) = H(\mathbf{p}, \mathbf{q}) - H(\mathbf{p}) = -\sum_i p_i \log_2 \frac{q_i}{p_i} \geq 0$$

KL divergence is always non-negative, with equality iff $\mathbf{p} = \mathbf{q}$. It is not symmetric. Minimising cross-entropy in classification is equivalent to minimising KL divergence (since $H(\mathbf{p})$ is fixed for one-hot labels).

We have already seen KL divergence in the VAE loss (Lecture 9), where it pushes the encoder's latent distribution towards $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

```python
import torch, torch.nn.functional as F

def entropy(p, eps=1e-10):
    return -(p * torch.log2(p + eps)).sum()

def cross_entropy_bits(p, q, eps=1e-10):
    return -(p * torch.log2(q + eps)).sum()

def kl_div(p, q, eps=1e-10):
    return cross_entropy_bits(p, q, eps) - entropy(p, eps)

# Weather example
p = torch.tensor([0.75, 0.25])
q = torch.tensor([0.50, 0.50])   # model assumes equal probabilities

print(f'H(p):        {entropy(p):.3f} bits')
print(f'H(p,q):      {cross_entropy_bits(p,q):.3f} bits')
print(f'D_KL(p||q):  {kl_div(p,q):.3f} bits wasted')

# VAE Gaussian KL (closed form)
mu, log_var = torch.tensor([0.5, -0.3]), torch.tensor([-0.5, -0.2])
kl_gaussian = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()
print(f'VAE KL term: {kl_gaussian:.4f}')
```

---

## 2  Self-Supervised Learning

### 2.1  The Labelling Bottleneck

ImageNet has 1.2M labelled images — years of human effort. The internet has billions of unlabelled images. Self-supervised learning (SSL) generates supervision from the raw data itself via **pretext tasks** whose labels can be derived automatically.

We care not about pretext task performance but about the **quality of learned representations** on downstream tasks.

### 2.2  Pretext Tasks

| Task | Label source | What the model learns |
|---|---|---|
| Rotation prediction | Rotate by 0°/90°/180°/270°, predict which | Image semantics (to distinguish orientations) |
| Relative patch location | Extract two patches, predict relative position | Object parts and structure |
| Jigsaw puzzle | Scramble patches, predict permutation | Spatial coherence, object structure |
| Inpainting | Mask a region, predict pixels | Scene context |
| Colourisation | Convert to greyscale, predict colours | Object-level semantics |

**General principle**: harder pretext tasks require deeper semantic understanding → better features.

### 2.3  Masked Autoencoders (MAE)

MAE applies BERT-style masking to ViT: mask 75% of image patches, train to reconstruct missing patches from visible ones. The high masking ratio forces genuine semantic understanding — simple texture interpolation cannot fill in large missing regions.

**Efficiency trick**: only the visible ~25% of tokens are processed by the heavy encoder (no mask tokens in the encoder). A lightweight decoder reconstructs all patches. Training is very fast: the encoder processes 1/4 of tokens per image.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def random_masking(n_patches, mask_ratio=0.75):
    n_visible = int(n_patches * (1 - mask_ratio))
    shuffle   = torch.randperm(n_patches)
    return shuffle[:n_visible].sort().values, shuffle[n_visible:].sort().values

visible_idx, masked_idx = random_masking(196, mask_ratio=0.75)
print(f'Visible: {len(visible_idx)}/196  Masked: {len(masked_idx)}/196')
# 49 visible, 147 masked — encoder processes only 49 tokens!

# MAE training objective: MSE on masked patches only
def mae_loss(pred, target, masked_idx):
    return F.mse_loss(pred[:, masked_idx], target[:, masked_idx])
```

---

## 3  Contrastive Representation Learning

### 3.1  The Core Idea

Learn embeddings where similar inputs are close and dissimilar inputs are far apart. Unlike pretext tasks, no task-specific knowledge is baked in — similarity is defined by data augmentation.

**Collapse problem**: the trivial solution is to map all inputs to the same point. Every contrastive method needs a collapse prevention mechanism.

### 3.2  Triplet Loss

Push anchor $\mathbf{x}$ closer to positive $\mathbf{x}^+$ (same class) than to negative $\mathbf{x}^-$ (different class), with margin $\epsilon$:

$$L_\text{triplet} = \max\!\left(0,\; d(\mathbf{x}, \mathbf{x}^+) - d(\mathbf{x}, \mathbf{x}^-) + \epsilon\right)$$

The margin + max prevents the trivial solution of collapsing all distances to zero.

### 3.3  NT-Xent Loss (SimCLR)

SimCLR (Chen et al., 2020) creates positive pairs via data augmentation and uses all other examples in the batch as negatives. For a batch of $N$ images, two augmented views create $2N$ images:

$$L = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

This is exactly **cross-entropy on a $(2N-1)$-way classification problem**: find the positive pair among all $2N-1$ other images. The connection to Section 1 is elegant.

SimCLR key findings: (1) augmentation composition is crucial (colour distortion + random crop); (2) a projection MLP head improves representation quality; (3) larger batches provide more negatives and improve performance.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T

def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, d)
    sim = torch.mm(z, z.t()) / temperature               # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)])
    return F.cross_entropy(sim, labels)

class SimCLR(nn.Module):
    def __init__(self, encoder, d_enc, d_proj=128):
        super().__init__()
        self.encoder   = encoder
        self.projector = nn.Sequential(
            nn.Linear(d_enc, d_enc), nn.ReLU(),
            nn.Linear(d_enc, d_proj))

    def forward(self, x1, x2):
        h1, h2 = self.encoder(x1), self.encoder(x2)
        return h1, h2, self.projector(h1), self.projector(h2)

backbone = models.resnet18(weights=None)
backbone.fc = nn.Identity()
model = SimCLR(backbone, 512)

x1, x2 = torch.randn(32, 3, 32, 32), torch.randn(32, 3, 32, 32)
_, _, z1, z2 = model(x1, x2)
loss = nt_xent_loss(z1, z2)
print(f'SimCLR loss: {loss.item():.4f}  (ideal: log({2*32-1}) ≈ {torch.log(torch.tensor(63.0)):.2f})')
```

---

## 4  Diffusion Models

### 4.1  Motivation

GANs generate in one pass — fast but unstable. VAEs generate in one pass — stable but blurry. Diffusion models spread generation across $T=1{,}000$ small, easy denoising steps. Each step is individually trivial; the cumulative effect produces the highest-quality images of any generative model.

### 4.2  Forward Process

A fixed Markov chain that gradually adds Gaussian noise over $T$ steps:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})$$

The noise schedule $0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1$ is fixed (not learned). After $T$ steps, $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Single-step shortcut**: defining $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\; (1-\bar\alpha_t)\mathbf{I})$$

$$\Rightarrow \quad \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$

This allows sampling any $t$ without simulating the full chain — essential for efficient training.

### 4.3  Reverse Process and Training Objective

The reverse process is a learned Gaussian:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I})$$

Rather than predicting the mean directly, the network $\boldsymbol{\epsilon}_\theta$ predicts the **noise** that was added. The training loss is simply MSE between predicted and actual noise:

$$L = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},\; t)\|^2\right]$$

The U-Net denoiser takes the noisy image and the time step $t$ (as a sinusoidal embedding) as inputs.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

class NoiseSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        betas          = torch.linspace(beta_start, beta_end, T)
        alphas         = 1 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)
        self.T         = T
        self.betas     = betas
        self.alphas    = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_ab   = alpha_bar.sqrt()
        self.sqrt_1mab = (1 - alpha_bar).sqrt()

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        s  = self.sqrt_ab[t].view(-1,1,1,1)
        sm = self.sqrt_1mab[t].view(-1,1,1,1)
        return s * x0 + sm * noise, noise

def ddpm_train_step(model, sched, x0, opt):
    B = x0.size(0)
    t     = torch.randint(0, sched.T, (B,))
    noise = torch.randn_like(x0)
    xt, _ = sched.q_sample(x0, t, noise)
    loss  = F.mse_loss(model(xt, t), noise)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def ddpm_sample(model, sched, shape):
    x = torch.randn(shape)
    for t_val in reversed(range(sched.T)):
        t = torch.full((shape[0],), t_val, dtype=torch.long)
        eps_pred = model(x, t)
        alpha  = sched.alphas[t_val]
        ab     = sched.alpha_bar[t_val]
        beta   = sched.betas[t_val]
        mean   = (x - beta / (1-ab).sqrt() * eps_pred) / alpha.sqrt()
        x = mean + (beta.sqrt() * torch.randn_like(x) if t_val > 0 else 0)
    return x.clamp(-1, 1)

sched = NoiseSchedule()
x0 = torch.randn(4, 3, 32, 32)
t  = torch.randint(0, 1000, (4,))
xt, eps = sched.q_sample(x0, t)
print(f'x0: {x0.shape}  xt: {xt.shape}  eps: {eps.shape}')  # all (4,3,32,32)
```

---

## 5  Conditional Diffusion and CFG

**Classifier-free guidance** (CFG) trains a single model to be both conditional and unconditional: during training, randomly drop the conditioning signal. At inference, combine both predictions:

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing))$$

Scale $s > 1$ amplifies conditioning; $s=1$ gives pure conditional prediction; $s=0$ gives unconditional. Stable Diffusion, DALL-E 2, and Imagen all use CFG.

---

## 6  CLIP: Contrastive Language-Image Pre-training

CLIP trains an image encoder and text encoder jointly on 400M image-text pairs from the internet using a symmetric contrastive loss (NT-Xent across modalities):

$$L_\text{CLIP} = \text{NT-Xent}(\{\mathbf{z}_\text{image}\}, \{\mathbf{z}_\text{text}\})$$

For each batch of $N$ pairs, matching image-text pairs should be close; non-matching pairs should be far apart.

**Zero-shot classification**: embed all class name prompts ("a photo of a [class]") with the text encoder. Classify a test image by finding the class whose text embedding has the highest cosine similarity to the image embedding. Achieves ~76% top-1 on ImageNet with no ImageNet training.

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(1/temperature)))

    def forward(self, img_emb, txt_emb):
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)
        temp = torch.exp(self.log_temp)
        logits = img @ txt.t() * temp        # (B, B)
        labels = torch.arange(img.size(0))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

clip_loss = CLIPLoss()
img_emb = torch.randn(32, 512)
txt_emb = torch.randn(32, 512)
print(f'CLIP loss: {clip_loss(img_emb, txt_emb).item():.4f}')

# Using pre-trained CLIP (HuggingFace):
# from transformers import CLIPModel, CLIPProcessor
# model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
# processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
# inputs = processor(text=['a cat','a dog'], images=pil_image, return_tensors='pt')
# outputs = model(**inputs)
# probs = outputs.logits_per_image.softmax(dim=1)
```

---

## 7  Latent Diffusion Models (Stable Diffusion)

Pixel-space diffusion on 512×512 images is expensive: 786,432 values per image, processed by a U-Net for $T=1{,}000$ steps. Latent Diffusion Models (Rombach et al., 2022) solve this by:

1. **Train a VAE**: compress images from $512 \times 512 \times 3$ to $64 \times 64 \times 4$ (compression factor ≈ 48×)
2. **Apply diffusion in latent space**: the U-Net operates on $64 \times 64 \times 4$ — much cheaper
3. **Condition on text via cross-attention**: text prompt encoded with CLIP's text encoder; U-Net's intermediate layers attend to text embeddings at each denoising step

Result: Stable Diffusion generates 512×512 images in seconds on consumer hardware, with quality comparable to or better than pixel-space models.

---

## 8  Summary

| Topic | Core idea | Key innovation |
|---|---|---|
| Cross-entropy | Measure distribution mismatch | Information-theoretic grounding for classification loss |
| SSL pretext tasks | Auto-generated labels from raw data | No human annotation needed |
| MAE | Reconstruct 75% masked patches | Encoder sees only 25% of tokens — very fast |
| SimCLR | Pull augmented pairs together, push others apart | NT-Xent = N-way cross-entropy; augmentation creates pairs |
| DDPM | Iterative denoising over $T$ steps | Noise prediction + single-step forward; stable training |
| CFG | Guide diffusion without separate classifier | Dual forward pass (conditional + unconditional) |
| CLIP | Joint image-text embedding via contrastive learning | Zero-shot classification; shared vision-language space |
| Latent Diffusion | Diffusion in VAE latent space | 48× compression → fast, memory-efficient generation |

The lecture closes the loop on the entire course. Self-supervised learning removes the labelling bottleneck; contrastive learning provides a general representation learning framework; diffusion models achieve generation quality GANs could not match; CLIP unifies vision and language. Together, these techniques underpin essentially every state-of-the-art system in computer vision today.

## References

- Ho, J. et al. (2020). DDPM. NeurIPS.
- He, K. et al. (2021). MAE. CVPR 2022.
- Chen, T. et al. (2020). SimCLR. ICML.
- Radford, A. et al. (2021). CLIP. ICML.
- Rombach, R. et al. (2022). Latent Diffusion Models. CVPR.
- Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance.
- HuggingFace annotated diffusion: huggingface.co/blog/annotated-diffusion
