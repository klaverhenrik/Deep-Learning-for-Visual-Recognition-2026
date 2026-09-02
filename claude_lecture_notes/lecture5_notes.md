# Lecture 5
# Training ConvNets — Part 1

*Deep Learning for Visual Recognition · Aarhus University*

These notes cover the four foundational pillars of training deep networks well: choosing the right activation function, preprocessing the data, initialising the weights correctly, and stabilising training with batch normalisation. Each section explains the 'why' in depth and shows how the concept maps directly to PyTorch.

---

## 1  Introduction: The Training Pipeline

Having built a CNN architecture (Lecture 4), training it sounds straightforward: define a loss, run gradient descent. In practice, naïve choices lead to networks that fail to converge, converge to poor solutions, or take prohibitively long to train. This lecture covers four decisions — activation function, data preprocessing, weight initialisation, and batch normalisation — that collectively determine whether training proceeds healthily. Each one addresses a specific failure mode.

To frame what follows: the central enemy is unstable gradient flow. Gradients are the learning signal that propagates backwards from the loss through every layer. If they vanish (become too small) in early layers, those layers stop learning. If they explode (become too large), the loss diverges. Every technique in this lecture exists, in one way or another, to keep gradient magnitudes in a healthy range throughout training.

> **The mini-batch training loop.** Sample a batch of images → forward pass (compute activations and loss) → backward pass (compute gradients via backprop) → update weights. This repeats until convergence. Everything in this lecture affects how reliably that loop makes progress.

---

## 2  Activation Functions

Activation functions are the non-linearities placed after each linear layer. Without them, stacking linear layers collapses to a single linear transformation, and the network cannot learn non-linear decision boundaries. But not all activation functions are equally well-suited for deep networks.

### 2.1  Sigmoid — The Classic Choice and Its Problems

The sigmoid $\sigma(x) = 1/(1 + e^{-x})$ squashes any input to $(0, 1)$. It was the default activation for decades, partly because its output is interpretable as a probability. But it has three serious problems for hidden layers in deep networks:

**Problem 1: Saturated neurons kill the gradient.** When $|x|$ is large, the sigmoid is in its flat region where $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \approx 0$. In the backpropagation equation $\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l)})^T \boldsymbol{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$, every saturated neuron multiplies the gradient by $\sim 0$. In a deep network with many saturated neurons, the gradient shrinks exponentially towards the input layers — the vanishing gradient problem.

**Problem 2: Non-zero-centred outputs cause zig-zagging.** The sigmoid always outputs positive values. If the input $x$ to a neuron is always positive (as it will be if the previous layer used sigmoid), then the gradient $\partial L/\partial w_i = (\partial L/\partial z)(\partial z/\partial w_i) = (\partial L/\partial z) \cdot x_i$. Since $x_i > 0$ always, all weight gradients in that neuron have the same sign as the upstream gradient $\partial L/\partial z$. This means the gradient vector is always in a quadrant (all positive or all negative), never in a mixed-sign direction. To reach the optimal weight vector — which often requires moving in a mixed-sign direction — gradient descent must zig-zag, which is slow.

**Problem 3: Computational cost.** The $\exp()$ operation is more expensive than the comparisons and multiplications used by ReLU. This is a minor concern relative to the first two, but still worth noting in compute-intensive training runs.

### 2.2  Tanh — A Better Sigmoid

$\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$ squashes inputs to $(-1, 1)$. Its output is zero-centred, which eliminates the zig-zagging problem. The gradient of tanh is $1 - \tanh^2(x)$, which is larger than $\sigma'(x)$ for the same input. However, tanh still saturates when $|x|$ is large, so it still suffers from the vanishing gradient problem in very deep networks.

### 2.3  ReLU — The Modern Default

$\text{ReLU}(x) = \max(0, x)$ is the default activation in modern deep networks. It solves the saturation problem: for positive inputs, $\text{ReLU}'(x) = 1$ — a constant gradient that neither vanishes nor explodes. The AlexNet paper (Krizhevsky et al., 2012) reported $\sim 6\times$ faster convergence with ReLU compared to tanh. It is also trivially cheap to compute (one comparison) and to differentiate.

ReLU has two weaknesses. First, its output is not zero-centred (always $\geq 0$), so the zig-zagging problem still applies. Second, the dying ReLU problem: if a neuron's input is always negative, it always outputs 0 and receives no gradient — it can never recover. This is most likely to happen when the learning rate is too high, causing a large weight update that permanently pushes the neuron's pre-activation into negative territory.

### 2.4  Leaky ReLU — Fixing the Dead Neuron Problem

Leaky ReLU introduces a small slope $\alpha$ (typically 0.01) for negative inputs: $f(x) = \max(\alpha x, x)$. This ensures the gradient is never exactly zero, preventing neurons from dying permanently. Parametric ReLU (PReLU) makes $\alpha$ a learnable parameter per channel. ELU uses a smooth exponential for the negative region.

> **Practical recommendation.** Use ReLU in hidden layers — it is fast, does not saturate for positive inputs, and works well with proper initialisation. If you see dead neurons (indicated by zero activations for most inputs), switch to Leaky ReLU or reduce your learning rate. Never use sigmoid in hidden layers of deep networks. Reserve sigmoid for output layers when you need a probability (binary classification).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Comparing activation functions ────────────────────────────────────
x = torch.linspace(-3, 3, 100)

sigmoid_out  = torch.sigmoid(x)        # output always in (0, 1)
tanh_out     = torch.tanh(x)           # output in (-1, 1), zero-centred
relu_out     = F.relu(x)              # max(0, x) — no saturation for x>0
leaky_out    = F.leaky_relu(x, 0.01)  # small slope for x<0

# ── Gradients (derivative at x = 2.0 and x = -2.0) ───────────────────
for name, fn in [('sigmoid', torch.sigmoid), ('tanh', torch.tanh),
                 ('relu', F.relu), ('leaky_relu', lambda x: F.leaky_relu(x,0.01))]:
    for val in [2.0, -2.0]:
        x_t = torch.tensor([val], requires_grad=True)
        fn(x_t).backward()
        print(f'{name:12s}  x={val:5.1f}  gradient={x_t.grad.item():.4f}')

# sigmoid   x= 2.0  gradient=0.1049   ← already small
# sigmoid   x=-2.0  gradient=0.1049   ← same (symmetric)
# tanh      x= 2.0  gradient=0.0707   ← smaller than sigmoid here
# tanh      x=-2.0  gradient=0.0707
# relu      x= 2.0  gradient=1.0000   ← constant, never saturates
# relu      x=-2.0  gradient=0.0000   ← dead zone
# leaky_relu x= 2.0 gradient=1.0000
# leaky_relu x=-2.0 gradient=0.0100   ← small but non-zero

# ── Dead ReLU demonstration ───────────────────────────────────────────
torch.manual_seed(0)
layer = nn.Linear(10, 5)

# Force all pre-activations to be very negative by using a bad init
with torch.no_grad():
    layer.weight.fill_(-1.0)
    layer.bias.fill_(-5.0)

x_in = torch.randn(4, 10)   # positive inputs
out  = F.relu(layer(x_in))
print(f'Dead neurons: {(out == 0).all(dim=0).sum().item()} / {out.shape[1]}')
# All 5 neurons are dead — no gradient will flow back through them
```

*Code 1 – Comparing activation function gradients. Note that sigmoid at $x=2.0$ already has a gradient of only 0.10, while ReLU has a constant gradient of 1.0. The dead ReLU demonstration shows all neurons going to zero when pre-activations are negative — a realistic failure mode when learning rate is too high.*

---

## 3  Data Preprocessing

### 3.1  Why Zero-Centring Matters

Consider fitting a line through data points clustered far from the origin (e.g., all positive). A small change in the slope creates a large change in the prediction error for every data point, because each point is far from where the line pivots. The loss landscape becomes very steep — a ravine — and gradient descent oscillates wildly rather than making steady progress.

This connects directly to the zig-zagging problem from Section 2.1. If the inputs to a layer are all positive (because they are raw pixel values, say, or sigmoid outputs), all weight gradients have the same sign. Gradient descent can only move in an axis-aligned staircase pattern rather than straight toward the optimum. Zero-centring the data removes this constraint.

### 3.2  Mean Subtraction and Standardisation

The standard preprocessing pipeline for images:

- **Step 1 — Mean subtraction**: Subtract the per-channel mean computed over the training set. This centres the data around zero. Each pixel of each image now reflects how much it differs from the average, rather than its absolute intensity.
- **Step 2 — Normalisation (optional)**: Divide by the per-channel standard deviation. This ensures that different channels (or features) are on roughly the same scale, preventing any one dimension from dominating the gradient.

Different architectures use slightly different conventions. AlexNet subtracts a single mean image (shape $32 \times 32 \times 3$). VGGNet subtracts per-channel means (3 scalars). ResNet both subtracts per-channel mean and divides by per-channel std. The specific values can be found in the original papers or framework documentation.

> **Critical pitfall.** Compute normalisation statistics (mean, std) only on the training set. Apply those same statistics to the validation and test sets. Never compute statistics on the validation or test set — doing so is a form of data leakage. The test set must remain completely uncontaminated.

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Computing dataset statistics from the training set ONLY ──────────
raw_train = datasets.CIFAR10('.', train=True, download=True,
                              transform=transforms.ToTensor())
loader    = DataLoader(raw_train, batch_size=1000, shuffle=False)

# Accumulate mean and std across all training images
mean = torch.zeros(3)
std  = torch.zeros(3)
n_batches = 0
for imgs, _ in loader:
    mean += imgs.mean(dim=[0, 2, 3])   # mean per channel
    std  += imgs.std(dim=[0, 2, 3])
    n_batches += 1
mean /= n_batches
std  /= n_batches
print(f'CIFAR-10 mean: {mean.tolist()}')   # ≈ [0.491, 0.482, 0.447]
print(f'CIFAR-10 std:  {std.tolist()}')    # ≈ [0.247, 0.243, 0.261]

# ── Apply normalisation to train, val, AND test sets ──────────────────
# These are the ResNet ImageNet statistics — used when fine-tuning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

normalise = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),          # data augmentation on train only
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalise,                            # same stats for train and test
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),           # deterministic crop for test
    transforms.ToTensor(),
    normalise,                            # SAME statistics as training
])

# ── Quick check: after normalisation, data should be ~zero-centred ────
sample_batch = next(iter(DataLoader(
    datasets.CIFAR10('.', train=True, transform=train_transform), batch_size=256)))[0]
print(f'After normalisation — mean: {sample_batch.mean():.4f}, std: {sample_batch.std():.4f}')
# Should be close to 0.0 and 1.0
```

*Code 2 – Computing and applying normalisation. The key discipline: compute statistics on training data only, then apply them identically to all splits. When using a pre-trained model, always use the statistics from the dataset it was trained on (e.g. ImageNet).*

---

## 4  Weight Initialisation

### 4.1  Why Initialisation Matters

The weights determine the network's behaviour at the very first forward pass. In a deep network, the forward pass consists of repeated matrix multiplications:

$$\mathbf{y} = \mathbf{W}^{(L)} \mathbf{W}^{(L-1)} \cdots \mathbf{W}^{(2)} \mathbf{W}^{(1)} \mathbf{x}$$

If every weight matrix scales its input by more than 1, activations grow exponentially with depth — exploding activations. If every matrix scales by less than 1, activations shrink exponentially — vanishing activations. In either case, the gradients that flow back through the same matrices during backpropagation explode or vanish in the same way, making training impossible.

### 4.2  All-Zero Initialisation — What Not to Do

Setting all weights to zero seems like a neutral starting point, but it creates a symmetry problem: every neuron in a given layer computes the same output (zero for any input with zero-mean data), receives the same gradient during backpropagation, and updates identically. All neurons in the layer become forever identical — they learn the same feature, wasting the layer's representational capacity. Note that biases can safely be initialised to zero, because neurons are still distinguished by their weights.

### 4.3  Small Random Numbers — Better, but Not Enough

Initialising weights as $\mathbf{W} = 0.01 \times \mathcal{N}(0, 1)$ breaks symmetry: each neuron starts at a unique random point and will diverge during training. This works well for shallow networks. But for deep networks, multiplying by 0.01 means that each layer's activations are $100\times$ smaller than its inputs. After 6 layers, activations are $10^{-12}$ times the input magnitude — effectively zero. The gradient statistics show the same vanishing pattern.

Increasing the scale, e.g. $\mathbf{W} = 0.05 \times \mathcal{N}(0, 1)$, has the opposite effect: activations saturate in the flat regions of tanh or sigmoid, and again the gradient vanishes. The challenge is to hit the 'sweet spot' where neither explosion nor vanishing occurs.

### 4.4  The Sweet Spot: Variance Preservation

The insight behind principled initialisation is to keep the variance of activations constant across layers. If each layer's output has the same variance as its input, the signal neither grows nor shrinks as it propagates forward — and the same holds for gradients travelling backwards. The two conditions to aim for are:

$$\mathbb{E}\!\left[\mathbf{a}^{(l+1)}\right] = 0 \qquad \text{(zero mean)}$$

$$\text{Var}\!\left[\mathbf{a}^{(l+1)}\right] = \text{Var}\!\left[\mathbf{a}^{(l)}\right] \qquad \text{(constant variance)}$$

### 4.5  Xavier (Glorot) Initialisation — for tanh and sigmoid

For the tanh activation (approximately linear near zero), a derivation by Glorot & Bengio (2010) shows that the variance of the activations in layer $l+1$ is:

$$\text{Var}\!\left[\mathbf{a}^{(l+1)}\right] = s_l \cdot \text{Var}[\mathbf{W}^{(l)}] \cdot \text{Var}\!\left[\mathbf{a}^{(l)}\right]$$

where $s_l$ is the number of neurons in layer $l$ (the fan-in). To keep the variance constant we need $\text{Var}[\mathbf{W}^{(l)}] = 1/s_l$, which means initialising weights from $\mathcal{N}(0, 1/\sqrt{s_l})$. A symmetric version that also accounts for the fan-out uses $\text{Var}[\mathbf{W}^{(l)}] = 2/(s_l + s_{l+1})$. Both variants are known as Xavier or Glorot initialisation.

### 4.6  Kaiming (He) Initialisation — for ReLU

Xavier initialisation assumes the activation function is approximately linear near zero with symmetric outputs. ReLU violates this assumption: it zeros out half its inputs on average, halving the effective variance at each layer. He et al. (2015) derived the correct scale for ReLU:

$$\text{Var}[\mathbf{W}^{(l)}] = \frac{2}{s_l} \quad \to \quad \text{initialise from } \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{s_l}}\right)$$

The factor of 2 compensates for the fact that ReLU discards half the signal. In PyTorch this is `nn.init.kaiming_normal_` with `mode='fan_in'`. For Leaky ReLU with slope $\alpha$, the factor 2 is replaced by $2/(1+\alpha^2)$.

> **Rule of thumb.** For sigmoid or tanh activations: use Xavier (Glorot) initialisation. For ReLU or Leaky ReLU: use Kaiming (He) initialisation. Never initialise all weights to the same value. PyTorch's `nn.Linear` and `nn.Conv2d` use Kaiming uniform by default — a sensible choice for most networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Visualising activation statistics under different initialisations ──
torch.manual_seed(0)
N_LAYERS = 10
DIM      = 512

def forward_stats(init_fn, activation):
    """Run a forward pass and report mean/std of activations per layer."""
    x = torch.randn(256, DIM)
    for _ in range(N_LAYERS):
        W = torch.empty(DIM, DIM)
        init_fn(W)
        x = activation(x @ W)
    return x.mean().item(), x.std().item()

configs = [
    ('σ=0.01 + tanh',  lambda W: nn.init.normal_(W, std=0.01),  torch.tanh),
    ('σ=0.05 + tanh',  lambda W: nn.init.normal_(W, std=0.05),  torch.tanh),
    ('Xavier + tanh',  nn.init.xavier_normal_,                  torch.tanh),
    ('Kaiming + ReLU', nn.init.kaiming_normal_,                 F.relu),
    ('σ=0.01 + ReLU',  lambda W: nn.init.normal_(W, std=0.01),  F.relu),
]

print(f'{"Config":25s}  mean      std')
print('-' * 50)
for name, init, act in configs:
    mean, std = forward_stats(init, act)
    print(f'{name:25s}  {mean:+.4f}   {std:.4f}')

# Expected output (approximately):
# σ=0.01 + tanh          mean ≈ 0       std ≈ 0.0001  ← vanished
# σ=0.05 + tanh          mean ≈ 0       std ≈ 1.0 (saturated, flat tanh)
# Xavier + tanh          mean ≈ 0       std ≈ 0.8     ← healthy
# Kaiming + ReLU         mean ≈ 0.4     std ≈ 0.7     ← healthy
# σ=0.01 + ReLU          mean ≈ 0       std ≈ 0.0     ← vanished

# ── Using PyTorch's built-in initialisation ───────────────────────────
model = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

# Apply Kaiming initialisation to all Linear layers
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)

# Apply Xavier for tanh layers
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
```

*Code 3 – Visualising the effect of initialisation on activation statistics. Small $\sigma$ (0.01) causes vanishing; large $\sigma$ (0.05) causes saturation in tanh. Xavier gives healthy statistics for tanh; Kaiming gives healthy statistics for ReLU.*

```python
import torch
import torch.nn as nn

# ── Demonstrating exploding vs vanishing with raw matrix multiplications
# (no activation, to isolate the initialisation effect)
torch.manual_seed(42)
L = 50       # number of layers
D = 64       # dimension
x = torch.randn(1, D)

for scale, name in [(1.5, 'scale=1.5 (exploding)'),
                    (0.5, 'scale=0.5 (vanishing)'),
                    (1.0/D**0.5, 'scale=1/sqrt(D) (stable)')]:
    out = x.clone()
    for _ in range(L):
        W   = scale * torch.randn(D, D)
        out = out @ W
    print(f'{name}: output norm = {out.norm():.2e}')

# scale=1.5 (exploding):         output norm ≈ 1e+20  ← nan in practice
# scale=0.5 (vanishing):         output norm ≈ 1e-10  ← gradient dead
# scale=1/sqrt(D) (stable):      output norm ≈ 1e+00  ← healthy
```

*Code 4 – Isolating the weight scale effect over 50 layers. A scale of 1.5 explodes, 0.5 vanishes, and $1/\sqrt{D}$ (the Xavier principle) stays stable. This is the core intuition behind all principled initialisation schemes.*

---

## 5  Batch Normalisation

### 5.1  The Problem: Internal Covariate Shift

Even with careful initialisation, the distribution of activations in each layer drifts during training as the weights of preceding layers change. A weight update in layer $l$ changes the distribution of inputs seen by layer $l+1$, which then must adapt — a phenomenon called internal covariate shift. The practical effect is that later layers are always chasing a moving target, slowing convergence and increasing sensitivity to hyperparameter choices.

Batch normalisation (Ioffe & Szegedy, 2015) addresses this directly: instead of hoping that initialisation and careful learning rates keep activations in a healthy range, it enforces that healthy range as a hard constraint on every layer's output.

### 5.2  The Normalisation Step

Given a mini-batch of $N$ activations for a particular feature dimension $j$, batch normalisation computes the standardised value:

$$\hat{Z}_{ij} = \frac{X_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}$$

where $\mu_j = \frac{1}{N}\sum_i X_{ij}$ is the batch mean and $\sigma_j^2$ is the batch variance, both computed per feature dimension. The small constant $\varepsilon$ prevents division by zero. After normalisation, $\hat{Z}_j$ has mean 0 and variance 1 across the batch — directly enforcing the two rules of thumb from Section 4.4.

### 5.3  The Learnable Scale and Shift ($\gamma$ and $\beta$)

Simply clamping all activations to zero mean and unit variance would be too restrictive. For example, if we normalise the inputs to a sigmoid, they all fall in the approximately linear region $(-1, 1)$ — the sigmoid becomes effectively linear and the non-linearity is wasted. The network needs the freedom to represent any distribution that is useful for the task.

Batch norm adds two learnable parameters per feature dimension: $\gamma$ (scale) and $\beta$ (shift), applied after normalisation:

$$Y_{ij} = \gamma_j \cdot \hat{Z}_{ij} + \beta_j$$

If the network learns $\gamma_j = \sigma_j$ and $\beta_j = \mu_j$, the transformation becomes the identity $Y = X$ — so batch norm can never make things worse than without it; at most the network learns to undo the normalisation. In practice, the network settles on values of $\gamma$ and $\beta$ that produce the most informative activation distribution for each feature.

### 5.4  Training vs Inference: A Critical Distinction

During training, $\mu_j$ and $\sigma_j^2$ are computed from the current mini-batch. This introduces a subtle stochasticity — each example's normalised value depends on the other examples in the same batch — which acts as a mild regulariser.

During inference, we cannot use batch statistics (we may be processing a single image). Instead, PyTorch maintains running averages of $\mu$ and $\sigma^2$ during training (using exponential moving averages) and uses those fixed statistics at test time. This is why you must call `model.eval()` before inference: it switches batch norm from using batch statistics to using the stored running averages. Forgetting this is one of the most common bugs in PyTorch code.

### 5.5  Batch Norm in Convolutional Networks

For fully connected layers, batch norm computes one mean and variance per feature dimension (one per column of the activation matrix). For convolutional layers, the convention is different: one mean and variance per channel (not per spatial position). A 2D feature map of shape $(N, C, H, W)$ is normalised so that each of the $C$ channels has zero mean and unit variance, computed over the $N \times H \times W$ values in that channel across the batch.

$$\text{For FC:} \quad \mu, \sigma \text{ have shape } (1, D)$$

$$\text{For Conv:} \quad \mu, \sigma \text{ have shape } (1, C, 1, 1)$$

### 5.6  Benefits of Batch Normalisation

- Stabilises gradient flow — prevents vanishing and exploding gradients.
- Allows higher learning rates — activations stay in a healthy range, so larger steps are safe.
- Reduces sensitivity to initialisation — the network is more robust to poor initial weights.
- Acts as a regulariser — the batch-dependent normalisation adds noise that reduces overfitting, often reducing the need for dropout.
- Zero overhead at test time — the normalisation can be algebraically fused into the preceding linear layer, so inference is as fast as without batch norm.

> **Where to place batch norm.** The standard placement is after the linear/conv operation and before the activation function: Conv → BN → ReLU. Some architectures place it after the activation (BN after ReLU), and recent work suggests post-norm can sometimes work better, but Conv → BN → ReLU is the safe default.

```python
import torch
import torch.nn as nn

# ── Batch normalisation: mechanics and PyTorch API ────────────────────

# ── For fully connected layers: nn.BatchNorm1d ────────────────────────
# Input shape: (batch, features)
bn1d = nn.BatchNorm1d(num_features=64)   # one γ and β per feature

x_fc = torch.randn(32, 64)   # batch of 32, 64 features each
y_fc = bn1d(x_fc)
print(f'BN1d output: mean={y_fc.mean():.4f}, std={y_fc.std():.4f}')
# Should be approximately 0.0 and 1.0 (before γ,β are updated)

# ── For convolutional layers: nn.BatchNorm2d ──────────────────────────
# Input shape: (batch, channels, H, W)
# Normalises across batch AND spatial dimensions, per channel
bn2d = nn.BatchNorm2d(num_features=32)   # one γ and β per channel

x_conv = torch.randn(8, 32, 14, 14)   # 8 images, 32 channels, 14×14
y_conv = bn2d(x_conv)
print(f'BN2d output: mean={y_conv.mean():.4f}, std={y_conv.std():.4f}')

# ── Standard CNN block: Conv → BN → ReLU ─────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        # bias=False because BN has its own bias (β)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ── CRITICAL: training vs eval mode ──────────────────────────────────
model = nn.Sequential(
    ConvBNReLU(3, 32), ConvBNReLU(32, 64),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
)

x = torch.randn(4, 3, 32, 32)

# Training mode: uses batch statistics, updates running averages
model.train()
out_train = model(x)
print(f'Train mode output shape: {out_train.shape}')

# Eval mode: uses stored running averages (fixed, deterministic)
model.eval()
with torch.no_grad():
    out_eval = model(x)
print(f'Eval mode output shape:  {out_eval.shape}')

# These will differ if running averages haven't warmed up yet:
print(f'Max difference: {(out_train - out_eval).abs().max():.4f}')

# ── Inspecting the learnable parameters ───────────────────────────────
bn = nn.BatchNorm2d(8)
print(f'gamma (weight) shape: {bn.weight.shape}')  # (8,) — one per channel
print(f'beta  (bias)   shape: {bn.bias.shape}')    # (8,) — one per channel
print(f'Running mean   shape: {bn.running_mean.shape}')  # (8,) — not a param
```

*Code 5 – Batch normalisation in PyTorch. Note `bias=False` in the `Conv2d` when followed by BN, since BN's $\beta$ parameter already serves as a bias. The `model.train()` / `model.eval()` distinction is critical — forgetting to call `eval()` is one of the most common PyTorch bugs.*

```python
import torch
import torch.nn as nn

# ── Demonstrating that BN stabilises training ─────────────────────────
# We compare a deep network with and without batch normalisation

def build_deep_net(use_bn, depth=20, width=128):
    layers = []
    in_ch  = 3
    for _ in range(depth):
        layers.append(nn.Conv2d(in_ch, width, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(width))
        layers.append(nn.ReLU(inplace=True))
        in_ch = width
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 10)]
    return nn.Sequential(*layers)

torch.manual_seed(0)
X = torch.randn(16, 3, 32, 32)
y = torch.randint(0, 10, (16,))

for use_bn in [False, True]:
    model = build_deep_net(use_bn=use_bn)
    # Use a high learning rate to stress-test
    opt   = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for step in range(20):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        if torch.isnan(loss): losses.append(float('nan')); break
        loss.backward()
        # Gradient norm tells us if gradients are healthy
        grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
        opt.step()
        losses.append(loss.item())

    label = 'WITH BN' if use_bn else 'WITHOUT BN'
    converged = not any(torch.isnan(torch.tensor(losses)))
    print(f'{label}: final loss={losses[-1]:.3f}, converged={converged}')
# WITHOUT BN: likely NaN (diverged with lr=0.1)
# WITH BN:    converges stably
```

*Code 6 – Batch normalisation stabilises a 20-layer deep network at a high learning rate (0.1). Without BN, the network typically diverges to NaN. With BN, it converges stably. This demonstrates BN's core practical benefit.*

---

## 6  Putting It All Together

The four techniques in this lecture are not independent: they interact, and the choices you make in one area affect the others. Here is how to think about combining them:

- **Zero-centre your data first.** This ensures the inputs to the first layer are already well-behaved and removes the zig-zagging problem from the start.
- **Use Kaiming initialisation with ReLU (or Xavier with tanh/sigmoid).** This gives the network a healthy starting point where activations neither explode nor vanish before any training has occurred.
- **Add batch normalisation after every conv/FC layer (before the activation).** This continuously corrects for drift as the weights change during training, removing the need for extremely careful initialisation and allowing higher learning rates.
- **Use ReLU activations.** With BN handling normalisation, the saturation of sigmoid/tanh is no longer partially mitigated — the network will train much faster with ReLU.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── A well-configured CNN incorporating all four techniques ───────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 1. DATA PREPROCESSING: normalise with training-set statistics
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # zero-centre
])

class WellConfiguredCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 2. BATCH NORM: Conv → BN → ReLU pattern throughout
            # 4. ACTIVATION FUNCTION: ReLU everywhere
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),   # BN before activation
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        # 3. WEIGHT INITIALISATION: Kaiming for ReLU layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)   # γ = 1 at init
                nn.init.zeros_(m.bias)    # β = 0 at init

    def forward(self, x):
        return self.classifier(self.features(x))

model = WellConfiguredCNN(num_classes=10)

# Count parameters
n = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n:,}')

# Verify activation statistics before training (should be ~N(0,1))
model.eval()
x = torch.randn(64, 3, 32, 32)   # simulate normalised batch
with torch.no_grad():
    # Hook to capture intermediate activations
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn
    model.features[2].register_forward_hook(hook('relu1'))  # after first ReLU
    model.features[5].register_forward_hook(hook('relu2'))  # after second ReLU
    _ = model(x)

for name, act in activations.items():
    print(f'{name}: mean={act.mean():.3f}, std={act.std():.3f}')
# Both should show healthy statistics (~zero mean, ~unit std at init)
```

*Code 7 – A complete well-configured CNN combining all four techniques: normalised inputs, Kaiming initialisation, Conv→BN→ReLU blocks, and ReLU activations. The forward hook inspection confirms that activation statistics are healthy before any training begins.*

---

## 7  Summary

This lecture addressed four foundational decisions that determine whether a deep network trains reliably. The table below distils the key takeaways and their PyTorch mappings:

| Topic | Core problem solved | Recommendation | PyTorch |
|---|---|---|---|
| Activation function | Non-linearity without gradient death or saturation | ReLU in hidden layers; sigmoid only at output for probability | `nn.ReLU()` |
| Data preprocessing | Zig-zagging gradients from non-zero-centred inputs | Subtract per-channel mean; divide by per-channel std | `transforms.Normalize(mean, std)` |
| Weight initialisation | Exploding/vanishing activations at network init | Kaiming for ReLU; Xavier for tanh/sigmoid; never all-zeros | `nn.init.kaiming_normal_()` |
| Batch normalisation | Internal covariate shift; slow convergence | Always use; insert after conv/FC, before activation | `nn.BatchNorm2d` / `BatchNorm1d` |

The overarching theme is gradient health: every technique in this lecture exists to keep the gradient signal flowing cleanly through the network during backpropagation. Batch normalisation is the single biggest practical win — it is so effective that modern deep learning practitioners treat it as the default rather than an optional extra. Lecture 6 picks up from here with the optimisation algorithms (SGD with momentum, Adam) and regularisation strategies (dropout, data augmentation) that round out the training recipe.

---

## References

- Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.
- He, K. et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV.
- Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.
- CS231n Stanford notes: cs231n.github.io/neural-networks-2/
- PyTorch documentation: pytorch.org/docs/stable/nn.init.html