# Lecture 6
# Training ConvNets — Part 2

*Deep Learning for Visual Recognition · Aarhus University*

These notes cover the second half of the training recipe: how to choose and tune an optimiser, prevent overfitting with regularisation, search for hyperparameters systematically, and diagnose problems by reading loss curves.

---

## 1  Optimisers: From SGD to Adam

### 1.1  Stochastic Gradient Descent Revisited

Gradient descent updates weights in the direction of steepest descent of the loss:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla J(\mathbf{w})$$

In full-batch gradient descent this gradient is computed over the entire training set — expensive for large datasets. Stochastic Gradient Descent (SGD) computes the gradient over a small mini-batch of $n$ examples and applies the update immediately. This has three practical benefits: updates are frequent (many per epoch), the mini-batch noise helps escape sharp local minima, and GPU memory constraints make full-batch gradient computation infeasible anyway.

Typical batch sizes are powers of 2 (32, 64, 128, 256) to align with GPU memory architecture. Larger batches reduce gradient noise and stabilise training; smaller batches introduce more randomness, which can help avoid bad local optima but produces noisier loss curves.

### 1.2  Problems with Vanilla SGD

SGD works, but it has well-known failure modes that become more pronounced in high-dimensional loss landscapes:

- **Ravines**: The loss surface often curves far more steeply in one direction than another. SGD oscillates across the steep direction while making slow progress along the flat one — like a ball rolling side to side in a valley rather than straight to the bottom.
- **Saddle points**: In high-dimensional space, local minima are rare. Much more common are saddle points — where the surface slopes down in some directions and up in others — and the gradient is zero at both. SGD stalls at saddle points because there is no gradient signal to follow.
- **Learning rate sensitivity**: A single global learning rate is applied to every parameter equally. Some weights may need large updates while others need tiny adjustments — one learning rate cannot be optimal for all.

### 1.3  Momentum

Momentum addresses the ravine problem by accumulating a velocity vector that points in the direction of sustained gradient flow. Think of a ball rolling down a hill: it builds up speed in directions the gradient consistently pushes it, and the oscillations in cross-directions cancel out over time.

The update equations are:

$$\mathbf{v}_t = \gamma \cdot \mathbf{v}_{t-1} + \nabla J(\mathbf{w})$$

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \mathbf{v}_t$$

where $\gamma \in [0, 1)$ is the momentum coefficient (typically 0.9 or 0.99) and $\mathbf{v}$ is the velocity. At each step, the new velocity is a weighted sum of the previous velocity and the current gradient. In the steady state — if the gradient points consistently in the same direction — the velocity grows to $\nabla J / (1-\gamma)$, effectively amplifying the effective learning rate by a factor of $1/(1-\gamma)$ in that direction. Oscillating gradient components average out to zero, damping the ravine behaviour.

> **Nesterov Momentum — looking ahead.** Standard momentum computes the gradient at the current position, then adds velocity. Nesterov momentum (NAG) computes the gradient at the anticipated future position — where the velocity would carry us — before applying it:
> $$\mathbf{v}_t = \gamma \cdot \mathbf{v}_{t-1} + \nabla J(\mathbf{w} + \gamma \cdot \mathbf{v}_{t-1})$$
> $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \mathbf{v}_t$$
> The intuition: standard momentum is a ball that blindly follows the slope. Nesterov momentum is a smarter ball that looks ahead and slows down before it would overshoot. In practice, Nesterov typically converges slightly faster than standard momentum and is available in PyTorch via `nesterov=True`.

> **Practical recommendation.** Always use momentum with SGD. Set $\gamma = 0.9$ as the default. Start with a relatively high learning rate and decay it over time. SGD + Momentum is the workhorse of computer vision training — ResNets, VGGs, and most landmark architectures were trained this way.

### 1.4  Adaptive Learning Rate Methods

Rather than using a single global learning rate, adaptive methods maintain a separate effective learning rate for each parameter, adjusted based on the history of gradients for that parameter. This automatically gives large updates to rarely-updated parameters and small updates to frequently-updated ones.

**Adagrad** accumulates the sum of squared gradients $G_{t,i}$ for each parameter $i$:

$$G_{t,i} = G_{t-1,i} + g_{t,i}^2$$

$$w_{t+1,i} = w_{t,i} - \frac{\alpha}{\sqrt{G_{t,i} + \varepsilon}} \cdot g_{t,i}$$

Parameters with a large historical gradient (frequently updated, or in steep dimensions) get a smaller effective learning rate $\alpha/\sqrt{G}$; parameters with a small historical gradient get a larger effective rate. Weakness: $G$ only grows — it never shrinks. Late in training, $G$ becomes so large that the effective learning rate approaches zero and learning stops entirely, even for parameters that still need updating.

**RMSProp — fixing Adagrad's shrinking learning rate** replaces the cumulative sum of squared gradients with an exponentially decaying moving average:

$$G_{t,i} = \gamma \cdot G_{t-1,i} + (1-\gamma) \cdot g_{t,i}^2$$

$$w_{t+1,i} = w_{t,i} - \frac{\alpha}{\sqrt{G_{t,i} + \varepsilon}} \cdot g_{t,i}$$

The decay rate $\gamma$ (typically 0.9) ensures that old gradients are gradually forgotten. The denominator stays bounded, preventing the learning rate from shrinking to zero. RMSProp was proposed by Hinton in a Coursera lecture and works well in practice, particularly for recurrent networks.

**Adam — momentum + RMSProp.** Adam (Adaptive Moment Estimation, Kingma & Ba 2015) combines the ideas of momentum (first moment) and RMSProp (second moment) with a bias correction for the early-training warm-up period:

$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \qquad \text{[1st moment: gradient mean]}$$

$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2 \qquad \text{[2nd moment: gradient variance]}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \text{[bias correction]}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \text{[bias correction]}$$

$$w_{t+1} = w_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

The bias correction (dividing by $1-\beta^t$) compensates for the fact that $m$ and $v$ are initialised at zero and are biased towards zero in early training. Default settings: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\alpha = 10^{-3}$. Intuition: Adam behaves like a heavy ball with friction — it builds up speed on consistent gradients but is stabilised by the second moment, which prevents runaway updates in directions of high curvature.

> **SGD vs Adam in practice.** Adam is a good default: it is robust to hyperparameter choices and often works well with a constant learning rate. SGD + Momentum can achieve slightly better final accuracy than Adam, but requires more careful learning rate tuning and scheduling. For new projects, start with Adam (`lr=1e-3`). For final model training or when squeezing out maximum performance, switch to SGD + Momentum with a cosine decay schedule.

```python
import torch
import torch.nn as nn

model  = nn.Linear(100, 10)   # toy model
loss_fn = nn.CrossEntropyLoss()

# ── SGD variants ──────────────────────────────────────────────────────
# Vanilla SGD
opt_sgd = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD + Momentum (γ=0.9 is the standard default)
opt_sgd_m = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD + Nesterov momentum
opt_nag = torch.optim.SGD(model.parameters(), lr=0.01,
                           momentum=0.9, nesterov=True)

# SGD + Momentum + Weight decay (L2 regularisation built in)
opt_wd = torch.optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4)

# ── Adaptive methods ──────────────────────────────────────────────────
# Adagrad
opt_adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)

# RMSProp (α=0.99 matches the γ decay rate from the slides)
opt_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adam — the usual default choice
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3,
                             betas=(0.9, 0.999), eps=1e-8)

# AdamW — Adam with proper weight decay (decoupled from gradient scaling)
# Preferred over Adam + weight_decay for most modern architectures
opt_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# ── Standard training step (same regardless of optimiser) ─────────────
x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))

opt_adam.zero_grad()              # 1. clear accumulated gradients
logits = model(x)                 # 2. forward pass
loss   = loss_fn(logits, y)       # 3. compute loss
loss.backward()                   # 4. backward pass
opt_adam.step()                   # 5. update weights
print(f'Loss: {loss.item():.4f}')
```

*Code 1 – All major PyTorch optimisers. Note AdamW: it applies weight decay directly to the weights rather than adding it to the gradient, which is theoretically cleaner and empirically better than Adam + `weight_decay`. AdamW is the default choice for Transformers and most modern architectures.*

---

## 2  Learning Rate Scheduling

The learning rate is the single most important hyperparameter. A good strategy is to start high to explore the loss landscape quickly, then decay to fine-tune into a precise local minimum. Too-slow decay wastes computation on chaotic exploration; too-aggressive decay freezes the model before it finds a good solution.

### 2.1  Step Decay

Multiply the learning rate by a fixed factor (e.g. 0.1) every $N$ epochs. Straightforward, but requires manual choice of decay epochs. A data-driven variant watches the validation loss and decays whenever it plateaus — this is PyTorch's `ReduceLROnPlateau`.

### 2.2  Cosine Annealing

Smoothly decays the learning rate following a cosine curve from $\alpha_0$ to near zero over $T$ steps:

$$\alpha_t = \frac{\alpha_0}{2} \cdot \left(1 + \cos\!\frac{\pi t}{T}\right)$$

Cosine annealing is widely used because it is smooth (no discontinuous jumps), reaches near-zero by the end, and works well empirically. `CosineAnnealingLR` in PyTorch implements this; `CosineAnnealingWarmRestarts` adds periodic restarts.

### 2.3  Linear Warmup

When training from scratch with a large batch size, starting with the full learning rate can cause early instability — the model may latch onto spurious patterns in the first few batches and spend the rest of training unlearning them. Linear warmup linearly increases the learning rate from 0 (or a small value) to the target rate over the first few epochs, then decays normally. This technique became widely used after the 'Training ImageNet in 1 Hour' paper (Goyal et al., 2017) showed it was necessary for stable large-batch training.

### 2.4  Cyclical Learning Rates

Rather than monotonically decaying, Cyclical Learning Rates (CLR) periodically oscillate the learning rate between a lower and upper bound. The rationale: occasionally pushing the learning rate back up forces the model to escape sharp, narrow minima and settle into flatter, more general ones. Combined with periodic restarts (warm restarts / SGDR), the model explores multiple basins of the loss landscape, and the best snapshot from any cycle can be kept.

```python
import torch
import torch.nn as nn

model     = nn.Linear(100, 10)
optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# ── Step decay: halve lr every 30 epochs ─────────────────────────────
scheduler_step = torch.optim.lr_scheduler.StepLR(
    optimiser, step_size=30, gamma=0.5)

# ── Cosine annealing: smooth decay over 100 epochs ───────────────────
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=100, eta_min=1e-6)

# ── Cosine annealing with warm restarts (cycles of length T_0) ───────
scheduler_wr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimiser, T_0=20, T_mult=2)   # restarts at epochs 20, 60, 140, …

# ── ReduceLROnPlateau: decay when val loss stops improving ────────────
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5, patience=5, verbose=True)

# ── Linear warmup + cosine decay (common in modern training) ─────────
def linear_warmup_cosine(optimiser, warmup_epochs, total_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

scheduler_warmup = linear_warmup_cosine(optimiser,
    warmup_epochs=5, total_epochs=100, base_lr=0.1)

# ── How to use a scheduler in a training loop ─────────────────────────
for epoch in range(100):
    # ... training code ...
    val_loss = 0.5   # placeholder

    # Step schedulers call .step() once per epoch
    scheduler_cos.step()

    # ReduceLROnPlateau needs the monitored metric
    scheduler_plateau.step(val_loss)

    if epoch % 20 == 0:
        current_lr = optimiser.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}  lr={current_lr:.6f}')
```

*Code 2 – Learning rate schedulers in PyTorch. The `linear_warmup_cosine` function implements the schedule used by most modern large-scale training runs. `ReduceLROnPlateau` is the simplest adaptive schedule: just watch `val_loss` and decay when it stops improving.*

---

## 3  Regularisation

Regularisation refers to any technique that reduces the gap between training and validation performance — i.e. that reduces overfitting. The central challenge is that neural networks have enormous capacity: given enough parameters, a network can memorise the training set. Regularisation adds inductive biases or constraints that prevent memorisation and force the network to learn generalisable patterns.

### 3.1  Early Stopping

Early stopping is the simplest and most universally applicable regularisation technique: monitor validation loss during training and stop (or save the best checkpoint) when it starts rising. The key idea is that the optimal stopping point — where training loss is still declining but validation loss is at its minimum — corresponds to the model with the best generalisation. Continuing to train beyond this point just memorises the training set.

In practice, rather than stopping the moment validation loss ticks up (which can be noisy), wait for `patience` epochs without improvement. Always save the best model checkpoint rather than using the final model.

### 3.2  Weight Decay (L2 Regularisation)

Weight decay adds a penalty proportional to the squared magnitude of the weights to the loss:

$$J_\text{reg}(\mathbf{w}) = J(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

This encourages the optimiser to prefer small weights, which biases the model towards simpler, smoother solutions. The gradient update gains an extra shrinkage term: $\mathbf{w} \leftarrow \mathbf{w}(1 - \alpha\lambda) - \alpha \cdot \partial J/\partial \mathbf{w}$. This is why it is called weight decay — the weights are multiplied by a factor slightly less than 1 at every step. In PyTorch, weight decay is passed directly to the optimiser via the `weight_decay` argument. Note that AdamW is preferred over Adam with `weight_decay`, because Adam + `weight_decay` scales the penalty by the gradient history, which is theoretically incorrect.

### 3.3  Dropout

Dropout (Srivastava et al., 2014) randomly sets a fraction $p$ of neuron activations to zero during each forward pass of training. With $p = 0.5$ (the common default), half the neurons are zeroed at each step. The effect is dramatic: the network can no longer rely on any single neuron — it is forced to develop redundant, distributed representations of each feature. This is powerful regularisation.

An alternative interpretation is that dropout trains an implicit ensemble of $2^n$ sub-networks simultaneously (one for each possible dropout mask), where $n$ is the number of neurons. At test time, all neurons are active, which approximately averages the predictions of all sub-networks.

**Scaling at test time.** If neurons are randomly zeroed with probability $p$ during training, the expected output of each neuron is $(1-p)$ times what it would be with all neurons active. At test time, all neurons are active, so the expected output is larger by a factor of $1/(1-p)$. Without correction, the test-time activations would systematically be larger than training-time activations, breaking the network. Inverted dropout (PyTorch's default) corrects for this during training: active neurons are scaled by $1/(1-p)$, so no adjustment is needed at test time.

Dropout is most effective in large fully connected layers. It is less commonly used in convolutional layers, because the spatial correlation between adjacent pixels means adjacent neurons are highly correlated, weakening the independence assumption that makes dropout effective. `Dropout2d` zeros entire feature map channels rather than individual units, which is more appropriate for convolutional features.

### 3.4  Data Augmentation

The most powerful regularisation technique available is simply training on more data. Data augmentation creates transformed versions of training images, effectively enlarging the training set without collecting new labels. The key principle: apply transformations that preserve the semantic content and class label.

Standard augmentations for image classification:

- Random horizontal flip — valid for most natural images (not text, digits, or objects with handedness).
- Random crop — sample a sub-region and resize to the target resolution. Forces the network to work with objects at varying positions.
- Colour jitter — random changes to brightness, contrast, saturation, and hue. Reduces sensitivity to lighting conditions.
- Rotation and scaling — teaches scale and rotation invariance.
- Gaussian noise — adds zero-mean random noise to pixel values, regularising against high-frequency features.

**Advanced augmentation strategies:**

- **Cutout**: Randomly zero out a rectangular region of the input during training. Forces the network to classify using the remaining parts — teaches occlusion robustness.
- **Mixup**: Blend two training images and their labels linearly: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$. Encourages linear behaviour between training examples and reduces over-confident predictions.
- **Label smoothing**: Replace hard one-hot targets (0/1) with soft targets ($\varepsilon/(k-1)$ / $1-\varepsilon$). Prevents the model from becoming overconfident, which improves calibration and generalisation.

> **A note on augmentation during inference.** Augmentation is applied only to the training set. At test time, use deterministic transformations only (e.g. centre crop, not random crop). Test-time augmentation (TTA) — averaging predictions over multiple augmented versions of the test image — can improve accuracy but increases inference cost.

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# ── Dropout ───────────────────────────────────────────────────────────
# nn.Dropout: zeros individual units (for FC layers)
# nn.Dropout2d: zeros entire channels (for conv feature maps)

class RegularisedNet(nn.Module):
    def __init__(self, p_drop=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout2d(p=0.1),  # drop whole channels — mild conv regularisation
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(p=p_drop),  # p=0.5 drops half of FC activations
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = RegularisedNet(p_drop=0.5)

# CRITICAL: Dropout behaves differently at train and test time
model.train()   # dropout is active
x = torch.randn(4, 3, 32, 32)
out1 = model(x)
out2 = model(x)   # different output! random mask each time
print('Train mode — outputs differ:', not torch.allclose(out1, out2))

model.eval()    # dropout is disabled (all neurons active, scaled)
with torch.no_grad():
    out3 = model(x)
    out4 = model(x)   # same output — deterministic
print('Eval mode  — outputs same:  ', torch.allclose(out3, out4))

# ── Data augmentation ─────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training: aggressive augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,  # colour jitter
                           saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validation/test: deterministic only — no randomness
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Label smoothing ───────────────────────────────────────────────────
# Built into PyTorch's CrossEntropyLoss since version 1.10
smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # ε = 0.1

logits = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))
loss_hard   = nn.CrossEntropyLoss()(logits, targets)
loss_smooth = smooth_loss(logits, targets)
print(f'Hard targets: {loss_hard:.4f}   Smoothed: {loss_smooth:.4f}')

# ── Mixup (manual implementation) ────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    """Mix pairs of examples. Returns blended (x, y) for the batch."""
    lam    = torch.distributions.Beta(alpha, alpha).sample().item()
    perm   = torch.randperm(x.size(0))
    x_mix  = lam * x + (1 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    # Loss = lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
    return x_mix, y_a, y_b, lam

x_batch = torch.randn(16, 3, 32, 32)
y_batch = torch.randint(0, 10, (16,))
x_mix, ya, yb, lam = mixup_batch(x_batch, y_batch)
print(f'Mixup λ={lam:.3f} — blended input shape: {x_mix.shape}')
```

*Code 3 – Dropout, data augmentation, label smoothing, and Mixup. The most critical detail: Dropout produces different outputs each forward pass during training (stochastic), but is disabled at eval time (deterministic). `model.eval()` is not optional — forgetting it is one of the most common bugs in deep learning code.*

---

## 4  Hyperparameter Search

### 4.1  What to Tune and at What Scale

Neural networks have many hyperparameters. The most important ones — roughly in decreasing order of sensitivity — are:

- Learning rate — often the single most impactful choice. Search on a log scale.
- Learning rate decay schedule — when and how fast to reduce the learning rate.
- Regularisation strength (weight decay $\lambda$, dropout probability $p$) — log scale for $\lambda$.
- Batch size — usually fixed by GPU memory; larger is not always better.
- Architecture choices — number of layers, width, use of residual connections etc.

Always search learning rate and weight decay on a logarithmic scale — powers of 10 span many orders of magnitude that matter, while linear search would waste most budget in a narrow range. Typical starting points: $\text{lr} \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$, weight decay $\in \{10^{-4}, 10^{-5}, 0\}$.

### 4.2  Random Search vs Grid Search

For most problems, random search outperforms grid search (Bergstra & Bengio, 2012). The intuition: in grid search, if one hyperparameter is unimportant, you waste experiments varying it while repeating the same effective values of the important one. In random search, every experiment varies all parameters, so important parameters are explored more densely in the same budget. Use random search by default.

### 4.3  A Practical 6-Step Protocol

The following coarse-to-fine search strategy works well in practice:

- **Step 1 — Check initial loss**: Before any training, verify that the loss at initialisation matches the theoretical value (e.g. $\log(n_\text{classes}) \approx 2.3$ for 10 classes with cross-entropy). If it doesn't, there is a bug in the model or data pipeline.
- **Step 2 — Overfit a tiny dataset**: Train on 5–10 mini-batches with regularisation disabled (`weight_decay=0`, no dropout). The model must be able to reach near-zero training loss. If it cannot, the architecture has a bug or insufficient capacity. Do not proceed until this works.
- **Step 3 — Find a working learning rate**: With the full training set and small weight decay, sweep $\text{lr} \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$. Find the largest lr that causes the loss to decrease significantly within 100 iterations.
- **Step 4 — Coarse grid, 1–5 epochs**: Try 5–20 combinations of lr and weight\_decay around the values from step 3. This quickly eliminates clearly bad combinations.
- **Step 5 — Refine grid, 10–20 epochs**: Take the top models from step 4 and train longer without learning rate decay. Zoom in on the best region.
- **Step 6 — Read the loss curves**: Always visualise training and validation loss. The shape of the curves tells you what is wrong and what to try next.

> **The most common debugging insight.** If `val_loss > train_loss` by a large margin: overfitting — add regularisation, augmentation, or reduce model size. If both losses are high: underfitting — increase model capacity, reduce regularisation, or check for bugs. If training loss oscillates rather than decreasing: learning rate is too high. If training loss decreases but validation loss never does: check for data leakage or a systematic preprocessing difference between train and val.

```python
import torch
import torch.nn as nn
import random, math

# ── Step 1: Sanity check the initial loss ─────────────────────────────
n_classes = 10
model = nn.Linear(100, n_classes)
x = torch.randn(64, 100)
y = torch.randint(0, n_classes, (64,))

with torch.no_grad():
    logits = model(x)
    init_loss = nn.CrossEntropyLoss()(logits, y)

expected = math.log(n_classes)   # ≈ 2.303 for 10 classes
print(f'Initial loss: {init_loss:.3f}  (expected ≈ {expected:.3f})')
# If these differ by more than ~0.5, something is wrong

# ── Step 2: Verify the model CAN overfit a tiny dataset ───────────────
tiny_x = torch.randn(32, 100)
tiny_y = torch.randint(0, n_classes, (32,))
opt    = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for step in range(500):
    opt.zero_grad()
    loss = loss_fn(model(tiny_x), tiny_y)
    loss.backward()
    opt.step()

acc = (model(tiny_x).argmax(1) == tiny_y).float().mean()
print(f'Tiny dataset accuracy: {acc:.0%}')   # Should be ~100%

# ── Random hyperparameter search ──────────────────────────────────────
def random_search(n_trials=20, epochs=5):
    results = []
    for trial in range(n_trials):
        # Sample on log scale
        lr     = 10 ** random.uniform(-4, -1)   # 1e-4 to 1e-1
        wd     = 10 ** random.uniform(-6, -3)   # 1e-6 to 1e-3
        model  = nn.Linear(100, n_classes)
        opt    = torch.optim.SGD(model.parameters(),
                                  lr=lr, momentum=0.9, weight_decay=wd)
        # Simulate a few epochs of training
        losses = []
        for ep in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(tiny_x), tiny_y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        results.append({'lr': lr, 'wd': wd, 'final_loss': losses[-1]})

    # Sort by final loss and print top 5
    results.sort(key=lambda r: r['final_loss'])
    print('Top 5 configurations:')
    for r in results[:5]:
        print(f"  lr={r['lr']:.2e}  wd={r['wd']:.2e}  loss={r['final_loss']:.4f}")

random_search()
```

*Code 4 – Hyperparameter search protocol. Step 1 checks the initial loss matches theory ($\log(n_\text{classes}) \approx 2.3$). Step 2 verifies the model can overfit a tiny dataset — if this fails, there is a bug. The random search samples lr and wd on a log scale, which covers the space efficiently.*

---

## 5  Reading Loss Curves

Loss curves are the most important diagnostic tool during training. A few minutes inspecting them can save hours of wasted compute by identifying problems early. Below is a guide to the most common patterns and what they mean.

### 5.1  The Learning Rate

- Loss explodes or immediately diverges to NaN: Learning rate is too high. Reduce by $10\times$.
- Loss decreases very slowly or flatlines after a few steps: Learning rate is too low. Increase by $10\times$.
- Loss oscillates but trends downward: Learning rate is slightly too high. A small reduction or momentum will help.
- Loss decreases smoothly and steadily: Learning rate is in a good range.

### 5.2  Overfitting vs Underfitting

- Training loss low, validation loss much higher (and possibly rising): Overfitting. Add regularisation, augmentation, reduce model size, or get more data.
- Both losses are high and close together: Underfitting. Model lacks capacity or is under-trained. Increase model size, reduce regularisation, train longer.
- Validation loss tracks training loss but slightly higher: Healthy training. The gap represents the natural train-val difference.

### 5.3  Noise in the Curves

Loss curves are always noisy because each data point represents a mini-batch, not the full dataset. If the curves are very noisy, try increasing the batch size — larger batches produce more stable gradient estimates. Look at the overall trend, not individual fluctuations. Using a running average (smoothing) helps visually.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Training loop with loss curve monitoring ──────────────────────────
torch.manual_seed(0)
N = 2000
X = torch.randn(N, 50)
y = (X[:, :5].sum(dim=1) > 0).long()
train_ds, val_ds = random_split(TensorDataset(X, y), [1600, 400])

model   = nn.Sequential(nn.Linear(50,128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128,64), nn.ReLU(), nn.Linear(64,2))
opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val, best_epoch, best_state = float('inf'), 0, None

for epoch in range(50):
    # ── TRAIN ─────────────────────────────────────────────────────────
    model.train()
    epoch_loss = 0
    for xb, yb in DataLoader(train_ds, batch_size=64, shuffle=True):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * len(xb)
    train_loss = epoch_loss / len(train_ds)

    # ── VALIDATE ──────────────────────────────────────────────────────
    model.eval()   # disable dropout — ALWAYS do this before validation
    with torch.no_grad():
        xv, yv  = val_ds[:]
        val_out  = model(xv)
        val_loss = loss_fn(val_out, yv).item()
        val_acc  = (val_out.argmax(1) == yv).float().mean().item()

    # ── RECORD ────────────────────────────────────────────────────────
    current_lr = opt.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    # ── EARLY STOPPING ────────────────────────────────────────────────
    if val_loss < best_val:
        best_val, best_epoch = val_loss, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    sched.step()

    if epoch % 10 == 0 or epoch == 49:
        print(f'Ep {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f}',
              f'acc={val_acc:.2%} lr={current_lr:.2e}')

# Restore best weights before final evaluation
model.load_state_dict(best_state)
print(f'\nBest epoch: {best_epoch}  best val loss: {best_val:.4f}')

# ── Reading the curves: diagnostic logic ──────────────────────────────
train_losses = history['train_loss']
val_losses   = history['val_loss']

gap = val_losses[-1] - train_losses[-1]
if gap > 0.3:
    print('⚠  Large train/val gap → likely overfitting')
elif train_losses[-1] > 0.5:
    print('⚠  High training loss → likely underfitting')
else:
    print('✓  Training looks healthy')
```

*Code 5 – A complete training loop with loss monitoring, early stopping, and cosine learning rate scheduling. The diagnostic block at the end programmatically flags the most common training problems. Note: `model.eval()` before every validation pass is essential for correct behaviour with BatchNorm and Dropout.*

---

## 6  Monitoring Gradients and Activations

Beyond the loss curve, two additional diagnostics give early warning of training pathologies: activation histograms per layer and gradient norms per layer.

### 6.1  Activation Distributions

Healthy activations for ReLU layers should be roughly half zero (the negative half is clipped) and half positive values spread across a reasonable range. Warning signs:

- All activations $\approx 0$ across all inputs: vanishing activations, likely due to poor initialisation or saturation. Check initialisation and batch norm placement.
- Activations clustered at $\pm 1$ (for tanh): saturation. Reduce weight scale.
- Activations growing layer by layer: exploding activations. Add batch norm or reduce weight scale.

### 6.2  Gradient Norms

Healthy gradient norms should be roughly similar across layers. Warning signs:

- Gradient norm $\approx 0$ in early layers but normal in later layers: vanishing gradients. Add residual connections or batch norm.
- Gradient norm growing layer by layer (largest in first layer): exploding gradients. Add gradient clipping or reduce learning rate.
- Gradient norm suddenly spikes to infinity mid-training: a single bad batch. Gradient clipping prevents this from corrupting the weights.

```python
import torch
import torch.nn as nn

# ── Monitoring activations and gradients during training ──────────────
model = nn.Sequential(
    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
    nn.Linear(64,  32), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Linear(32,  10)
)

# ── Activation monitoring with forward hooks ──────────────────────────
activation_stats = {}

def make_activation_hook(name):
    def hook(module, input, output):
        act = output.detach()
        activation_stats[name] = {
            'mean':     act.mean().item(),
            'std':      act.std().item(),
            'frac_zero': (act == 0).float().mean().item(),  # fraction dead
        }
    return hook

# Register hooks on ReLU activations
model[2].register_forward_hook(make_activation_hook('relu1'))
model[5].register_forward_hook(make_activation_hook('relu2'))

x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))
loss = nn.CrossEntropyLoss()(model(x), y)

print('Activation statistics after one forward pass:')
for name, stats in activation_stats.items():
    print(f'  {name}: mean={stats["mean"]:+.3f}  std={stats["std"]:.3f}  dead={stats["frac_zero"]:.1%}')
# Healthy: mean ≈ 0.3–0.5, std ≈ 0.5–1.0, dead ≈ 30–50%

# ── Gradient monitoring with backward hooks ───────────────────────────
loss.backward()

print('\nGradient norms per layer:')
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f'  {name:25s}: {grad_norm:.4f}')

# ── Gradient clipping — prevents explosion from a single bad batch ────
optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss2 = nn.CrossEntropyLoss()(model(x), y)
loss2.backward()

# Clip gradient norm to 1.0 before the weight update
total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f'\nGradient norm before clipping: {total_grad_norm:.4f}')
optimiser.step()

# ── Weight visualisation (first layer filters) ────────────────────────
# Check that first-layer filters look like structured features (edges,
# blobs), not random noise. Noisy filters = training hasn't converged
# or learning rate is too low / weight decay too strong.
first_layer_weights = model[0].weight.detach()  # shape (64, 128)
print(f'First layer weight norm: {first_layer_weights.norm(dim=1).mean():.4f}')
```

*Code 6 – Activation and gradient monitoring using PyTorch hooks. Forward hooks capture activations without modifying the model; backward hooks (or `named_parameters()` iteration) show gradient norms. Gradient clipping (`clip_grad_norm_`) is a defensive measure that prevents a single bad batch from corrupting the model.*

---

## 7  The Complete Training Recipe

Combining everything from Lectures 5 and 6, here is the full checklist for training a deep CNN from scratch:

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# ════════════════════════════════════════════════════════════════════
# COMPLETE CNN TRAINING RECIPE
# Combines all techniques from Lectures 5 and 6
# ════════════════════════════════════════════════════════════════════

# 1. DATA PREPROCESSING ───────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),            # zero-centre
])
val_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

# 2. MODEL: Conv→BN→ReLU with Kaiming init ────────────────────────────
backbone = models.resnet18(weights=None)       # train from scratch
backbone.fc = nn.Linear(512, 10)               # 10-class head

for m in backbone.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)

# 3. OPTIMISER + WEIGHT DECAY ─────────────────────────────────────────
optimiser = torch.optim.SGD(backbone.parameters(),
    lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

# 4. LEARNING RATE SCHEDULE: warmup + cosine ──────────────────────────
warmup = torch.optim.lr_scheduler.LinearLR(
    optimiser, start_factor=0.01, total_iters=5)  # 5 warmup epochs
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=95, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimiser, schedulers=[warmup, cosine], milestones=[5])

# 5. LOSS: with label smoothing ───────────────────────────────────────
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# 6. TRAINING LOOP with early stopping ────────────────────────────────
best_val, patience_left = float('inf'), 10

for epoch in range(100):
    backbone.train()
    for x, y in DataLoader([]): pass   # real data loader here
        # opt.zero_grad(); loss = loss_fn(backbone(x), y)
        # loss.backward()
        # nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        # optimiser.step()

    backbone.eval()
    with torch.no_grad():
        pass  # compute val_loss on validation set
    val_loss = 0.5  # placeholder

    scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        patience_left = 10
        torch.save(backbone.state_dict(), 'best_model.pt')
    else:
        patience_left -= 1
        if patience_left == 0:
            print(f'Early stopping at epoch {epoch}')
            break

# 7. RESTORE BEST ─────────────────────────────────────────────────────
backbone.load_state_dict(torch.load('best_model.pt'))
```

*Code 7 – The complete training recipe combining all techniques from Lectures 5 and 6: zero-centred data, data augmentation, Conv→BN→ReLU with Kaiming init, SGD+Nesterov+weight\_decay, linear warmup + cosine annealing, label smoothing, gradient clipping, and early stopping.*

---

## 8  Summary

Lectures 5 and 6 together complete the training recipe for deep CNNs. The table below maps every major technique to its PyTorch equivalent and the problem it solves:

| Technique | Problem solved | PyTorch |
|---|---|---|
| Momentum | Ravine oscillation; saddle points | `SGD(momentum=0.9, nesterov=True)` |
| Adam / AdamW | Per-parameter adaptive learning rates | `torch.optim.Adam` / `AdamW(lr=1e-3)` |
| LR warmup | Early instability with large batch training | `LinearLR(start_factor=0.01)` |
| Cosine annealing | Sub-optimal final convergence | `CosineAnnealingLR(T_max=N)` |
| LR on plateau | Automatic decay when progress stalls | `ReduceLROnPlateau(patience=5)` |
| Early stopping | Overfitting — val loss starts rising | Save best checkpoint manually |
| Weight decay | Overfitting — model too complex | `weight_decay=1e-4` in optimiser |
| Dropout | Co-adaptation of neurons; overfitting | `nn.Dropout(p=0.5)` |
| Data augmentation | Limited training set; overfitting | `transforms.RandomResizedCrop`, etc. |
| Label smoothing | Overconfidence; brittle predictions | `CrossEntropyLoss(label_smoothing=0.1)` |
| Gradient clipping | Exploding gradients from bad batches | `clip_grad_norm_(params, max_norm=1.0)` |
| Loss curve reading | Diagnosing all training problems | Log and plot `train_loss`, `val_loss` |

The common thread running through every technique in this lecture is the bias-variance trade-off: complex models with many parameters have low bias (they can fit any pattern) but high variance (they overfit). Regularisation techniques — dropout, weight decay, augmentation, early stopping — all reduce variance, typically at the cost of a small increase in bias. The art of training is finding the sweet spot. With the techniques in Lectures 5 and 6, you have all the tools needed to do so systematically rather than by trial and error.

---

## References

- Kingma, D. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR.
- Smith, L. (2015). Cyclical Learning Rates for Training Neural Networks. WACV 2017.
- Goyal, P. et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv.
- Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. ruder.io/optimizing-gradient-descent/
- PyTorch optimiser docs: pytorch.org/docs/stable/optim.html