# Lecture 5 — Training ConvNets Part 1

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes cover the one-time setup decisions made before training begins: activation functions, data preprocessing, weight initialisation, and batch normalisation. The unifying theme throughout is **gradient health** — each topic addresses a specific way that gradients can become too small (vanishing), too large (exploding), or poorly conditioned, and proposes a fix.

---

## 1  Activation Functions

### 1.1  Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

**Three problems:**

1. **Saturating gradients**: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. For inputs with $|z| \gg 0$, the derivative approaches zero. Gradients propagating through many sigmoid layers are multiplied by these small values repeatedly, vanishing exponentially.

2. **Non-zero-centred outputs**: outputs are always positive $(0, 1)$. This means the gradients flowing into the previous layer are always the same sign, causing **zig-zag dynamics** in gradient descent — the weight updates can only all increase or all decrease together, slowing convergence.

3. **Expensive to compute**: requires `exp`.

### 1.2  Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$$

Zero-centred (fixes problem 2 of sigmoid) but still saturates (problem 1 remains). Preferred over sigmoid for hidden layers when a saturating activation is needed.

### 1.3  ReLU

$$\text{ReLU}(z) = \max(0, z)$$

**Advantages:**
- Does **not** saturate in the positive region: $\text{ReLU}'(z) = 1$ for $z > 0$
- Extremely cheap to compute (just a threshold)
- Converges approximately **6× faster** than sigmoid/tanh in practice

**One problem — Dead ReLU**: neurons where $z < 0$ have zero gradient and never update. Once a neuron produces negative pre-activations for all training examples, it is permanently silent. Caused by high learning rates or unlucky initialisation.

### 1.4  Leaky ReLU and Variants

$$\text{Leaky ReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha = 0.01$$

Fixes the dead neuron problem by allowing a small gradient ($\alpha$) when the unit is not active. **PReLU** makes $\alpha$ a learnable parameter per channel. **ELU** and **GELU** are other smooth alternatives used in Transformers.

**Default recommendation**: use **ReLU** for CNNs. Switch to Leaky ReLU if dead neurons are a problem (check activation maps during training — all-zero feature maps indicate dead filters).

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Activation function comparison ───────────────────────────────────
z = torch.linspace(-4, 4, 200)

activations = {
    'Sigmoid':     torch.sigmoid(z),
    'Tanh':        torch.tanh(z),
    'ReLU':        torch.relu(z),
    'Leaky ReLU':  nn.LeakyReLU(0.1)(z),
}

# Gradient magnitude at z=0 and z=3
for name, a in activations.items():
    z_input = torch.tensor([0.0, 3.0], requires_grad=True)
    out = nn.functional.relu(z_input) if 'ReLU' in name else torch.sigmoid(z_input)
    out.sum().backward()
    print(f'{name:12s}  grad at z=0: {z_input.grad[0]:.3f},  z=3: {z_input.grad[1]:.3f}')
    z_input.grad = None  # reset

# sigmoid:     grad at z=0: 0.250, z=3: 0.045  ← saturates
# tanh:        grad at z=0: 1.000, z=3: 0.010  ← saturates more severely
# relu:        grad at z=0: 0.000, z=3: 1.000  ← dead at z=0 but constant elsewhere
```

---

## 2  Data Preprocessing

### 2.1  Zero-Centring

Subtract the **per-channel mean** computed over the training set:

$$\tilde{x} = x - \mu_\text{train}$$

This centres the input distribution around zero, alleviating the non-zero-centred output problem of sigmoid and improving gradient flow in the first layer.

### 2.2  Normalisation

Optionally divide by the per-channel standard deviation:

$$\tilde{x} = \frac{x - \mu_\text{train}}{\sigma_\text{train}}$$

This ensures all input features have unit variance, preventing features with large magnitude from dominating the gradient.

**Critical rule**: compute $\mu$ and $\sigma$ on the **training set only**. Apply the same statistics to the validation and test sets. Never compute statistics on the test set — this leaks information.

### 2.3  Why Preprocessing Helps

If inputs are not zero-centred, the gradient update to the first-layer weights must have all the same sign (since the gradient $\partial J / \partial \mathbf{w} \propto \mathbf{x}$, and if all $x > 0$ then all weight gradients have the same sign as $\boldsymbol{\delta}$). This creates the zig-zag dynamics described in Section 1.1.

```python
import torch

# ── Correct preprocessing pipeline ───────────────────────────────────
# Compute statistics on training data ONLY
X_train = torch.randn(1000, 3, 32, 32)   # simulated CIFAR images
X_val   = torch.randn(200,  3, 32, 32)

# Per-channel mean and std over training set
mean = X_train.mean(dim=[0, 2, 3], keepdim=True)   # (1, 3, 1, 1)
std  = X_train.std( dim=[0, 2, 3], keepdim=True)   # (1, 3, 1, 1)

# Apply to both train and val (same statistics!)
X_train_norm = (X_train - mean) / (std + 1e-8)
X_val_norm   = (X_val   - mean) / (std + 1e-8)   # use TRAIN mean/std

print(f'Train mean after norm: {X_train_norm.mean([0,2,3]).tolist()}')  # ≈ [0,0,0]
print(f'Train std  after norm: {X_train_norm.std([0,2,3]).tolist()}')   # ≈ [1,1,1]

# In practice, use torchvision transforms:
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet means
#                          std= [0.229, 0.224, 0.225])   # ImageNet stds
# ])
```

---

## 3  Weight Initialisation

### 3.1  Why Initialisation Matters

Weights determine the initial scale of pre-activations $\mathbf{z}^{[\ell]}$. If weights are too large, $\mathbf{z}$ saturates the activation and gradients vanish. If too small, $\mathbf{z} \approx 0$ everywhere and the network learns nothing.

### 3.2  All-Zero Initialisation — Never Do This

If $\mathbf{w} = \mathbf{0}$, every neuron in a layer computes the same output, receives the same gradient, and updates identically. They remain identical forever: **symmetry is never broken**, and the layer is effectively a single neuron regardless of its width.

### 3.3  Xavier / Glorot Initialisation

For activations that are symmetric around zero (tanh, sigmoid), the variance of the pre-activations should be preserved across layers. If each weight $w \sim \mathcal{N}(0, \sigma^2)$ independently, then:

$$\text{Var}(z^{[\ell]}) = n^{[\ell-1]} \cdot \sigma^2 \cdot \text{Var}(a^{[\ell-1]})$$

Setting $\sigma^2 = 1/n_\text{in}$ keeps variance constant:

$$w \sim \mathcal{N}\!\left(0,\ \frac{1}{n_\text{in}}\right)$$

or equivalently the uniform variant $w \sim \mathcal{U}\!\left(-\sqrt{6/(n_\text{in}+n_\text{out})},\ +\sqrt{6/(n_\text{in}+n_\text{out})}\right)$.

### 3.4  Kaiming / He Initialisation

For **ReLU** activations, half the neurons are zeroed out on average, halving the effective variance. Xavier would result in the variance halving at each layer. The correction is:

$$w \sim \mathcal{N}\!\left(0,\ \frac{2}{n_\text{in}}\right)$$

The factor of 2 compensates for the half-zeroing effect of ReLU. This is the **default initialisation for modern CNN training**.

```python
import torch
import torch.nn as nn

# ── Why initialisation matters: variance through a deep network ───────
torch.manual_seed(0)
n_layers, n = 50, 256

def variance_trace(init_std):
    x = torch.randn(n)
    vars_ = [x.var().item()]
    for _ in range(n_layers):
        w = torch.randn(n, n) * init_std
        x = torch.relu(w @ x)
        vars_.append(x.var().item())
    return vars_

# Too small: variance collapses to zero
v_small  = variance_trace(0.01)
# Xavier: still collapses with ReLU (designed for tanh)
v_xavier = variance_trace((1/n)**0.5)
# Kaiming: variance stable
v_kaiming= variance_trace((2/n)**0.5)

print(f'Small init  — final var: {v_small[-1]:.2e}')    # → 0
print(f'Xavier init — final var: {v_xavier[-1]:.2e}')   # → 0 (wrong for ReLU)
print(f'Kaiming init— final var: {v_kaiming[-1]:.2e}')  # → ≈ 1

# PyTorch applies Kaiming uniform by default for nn.Linear and nn.Conv2d.
# Explicit control:
layer = nn.Linear(256, 256)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
nn.init.zeros_(layer.bias)
```

---

## 4  Batch Normalisation

### 4.1  The Problem: Internal Covariate Shift

During training, as the parameters of layer $\ell$ change, the distribution of inputs to layer $\ell+1$ also changes. This is called **internal covariate shift**. It forces each layer to constantly adapt to a shifting input distribution, slowing convergence.

### 4.2  The Solution

Batch normalisation (Ioffe & Szegedy, 2015) normalises the pre-activations of each layer to have zero mean and unit variance within each mini-batch, then applies a learnable affine transformation:

**Step 1 — Batch statistics:**

$$\mu_\mathcal{B} = \frac{1}{B} \sum_{i=1}^{B} z_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{B} \sum_{i=1}^{B} (z_i - \mu_\mathcal{B})^2$$

**Step 2 — Normalise:**

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

**Step 3 — Scale and shift (learnable):**

$$y_i = \gamma \hat{z}_i + \beta$$

The learnable parameters $\gamma$ and $\beta$ allow the network to undo the normalisation if that is what the data requires (e.g. if it is beneficial to have a non-zero mean or non-unit variance).

### 4.3  Train vs Inference

At **training time**: $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are computed from the current mini-batch.

At **inference time**: using a single test example's statistics would be noisy. Instead, BatchNorm maintains **running estimates** of $\mu$ and $\sigma^2$ during training (exponential moving average) and uses these fixed values at inference.

Always call `model.eval()` before running inference — this switches BatchNorm (and Dropout) to inference mode.

### 4.4  Where to Place BatchNorm

Standard placement: **after the linear transformation, before the activation**:

$$\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b} \;\to\; \text{BN}(\mathbf{z}) \;\to\; \text{ReLU}(\cdot)$$

For convolutional layers, BN is applied **per channel** (separate $\gamma$ and $\beta$ per output channel, statistics computed over the spatial and batch dimensions).

```python
import torch
import torch.nn as nn

# ── BatchNorm in a CNN block ──────────────────────────────────────────
block = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),     # per-channel BN for conv layers
    nn.ReLU(inplace=True),
)

x = torch.randn(16, 64, 14, 14)   # batch=16, C=64, 14×14

# Training mode: stats from this mini-batch
block.train()
out_train = block(x)
print(f'Train output shape: {out_train.shape}')  # (16, 128, 14, 14)

# Inference mode: stats from running estimates
block.eval()
with torch.no_grad():
    out_eval = block(x)
print(f'Eval  output shape: {out_eval.shape}')   # (16, 128, 14, 14)

# ── BN for FC layers ──────────────────────────────────────────────────
fc_block = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),     # per-feature BN for FC layers
    nn.ReLU(inplace=True),
)

# ── Inspecting running statistics ─────────────────────────────────────
bn = nn.BatchNorm2d(64)
x2 = torch.randn(32, 64, 8, 8)
bn.train()
_ = bn(x2)   # update running mean/var
print(f'Running mean (first 3 channels): {bn.running_mean[:3]}')
print(f'Running var  (first 3 channels): {bn.running_var[:3]}')
print(f'Learnable gamma (scale): {bn.weight[:3]}')   # initialised to 1
print(f'Learnable beta  (shift): {bn.bias[:3]}')     # initialised to 0
```

---

## 5  Putting It All Together

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ── Well-configured CNN for CIFAR-10 ─────────────────────────────────
transform = T.Compose([
    T.RandomCrop(32, padding=4),          # data augmentation
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], # CIFAR-10 train means
                [0.2023, 0.1994, 0.2010]),
])

class WellConfiguredCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Kaiming init applied automatically by PyTorch for Conv2d
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.MaxPool2d(2),                   # 32 → 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),                   # 16 → 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model = WellConfiguredCNN()

# Verify weight initialisation: Kaiming normal should give ≈unit variance
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        print(f'{name}: weight std = {m.weight.std():.3f}')
        break  # just check first conv
```

---

## 6  Summary

| Problem | Cause | Solution |
|---|---|---|
| Vanishing gradients | Sigmoid/tanh saturation | Use ReLU |
| Dead neurons | ReLU always negative | Kaiming init; lower LR; Leaky ReLU |
| Non-zero-centred inputs | Raw pixel values in [0,255] | Subtract channel mean |
| Varying feature scales | Different features at different scales | Divide by channel std |
| Symmetry (all weights equal) | Zero initialisation | Random init |
| Variance collapse/explosion | Wrong init scale | Xavier (tanh) or Kaiming (ReLU) |
| Internal covariate shift | Layer inputs shift as params change | Batch normalisation |

The right combination for a modern CNN is: **ReLU activations** + **Kaiming initialisation** + **input normalisation** + **Batch normalisation after each conv layer**. This is the standard recipe used in ResNets, VGGs, and most production models.

## References

- Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. ICML.
- He, K. et al. (2015). Delving Deep into Rectifiers (Kaiming init). ICCV.
- Glorot, X. & Bengio, Y. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks (Xavier init). AISTATS.
