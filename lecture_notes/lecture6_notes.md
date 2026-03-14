# Lecture 6 — Training ConvNets Part 2

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes cover the decisions made *during* training: optimisers beyond vanilla SGD, learning rate scheduling, regularisation techniques, and hyperparameter search. The closing section is a practical six-step training recipe that brings everything together.

---

## 1  Optimisers

### 1.1  Vanilla SGD and Its Problems

Standard (stochastic) gradient descent updates weights by:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$$

This has two practical problems:

1. **Oscillation in narrow valleys**: loss landscapes often have very different curvatures in different directions. SGD uses the same learning rate for all directions, causing it to oscillate across steep directions while crawling slowly along shallow ones.

2. **Local minima and saddle points**: vanilla SGD can get stuck at saddle points (far more common than true local minima in high-dimensional spaces) where the gradient is zero but the point is not optimal.

### 1.2  Momentum

Accumulate a velocity vector $\mathbf{v}$ that builds up in directions of persistent gradient:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \alpha \nabla J(\mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_{t+1}$$

Typical $\mu = 0.9$. Momentum accelerates convergence along consistent gradient directions, damps oscillation across inconsistent directions, and helps escape shallow local minima.

### 1.3  Nesterov Momentum

A small correction: evaluate the gradient at the "lookahead" position $\mathbf{w} + \mu \mathbf{v}$ rather than the current position:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \alpha \nabla J(\mathbf{w}_t + \mu \mathbf{v}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_{t+1}$$

This gives a more "anticipatory" correction and slightly improves convergence in practice.

### 1.4  AdaGrad

Maintain a per-parameter sum of squared gradients and divide the learning rate by its square root:

$$G_{t+1} = G_t + (\nabla J)^2 \quad \text{(element-wise)}$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{G_{t+1} + \epsilon}} \nabla J(\mathbf{w}_t)$$

Frequent parameters receive smaller updates; infrequent parameters receive larger ones. Useful for sparse data (NLP). Problem: $G$ only grows, so the learning rate eventually shrinks to zero.

### 1.5  RMSProp

Fix AdaGrad's monotonic decay by using an **exponential moving average** of squared gradients:

$$G_{t+1} = \rho G_t + (1-\rho)(\nabla J)^2$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{G_{t+1} + \epsilon}} \nabla J(\mathbf{w}_t)$$

Typical $\rho = 0.9$. The moving average "forgets" old gradients, keeping the effective learning rate from decaying to zero.

### 1.6  Adam

Combines momentum (first moment) and RMSProp (second moment) with bias correction:

$$m_{t+1} = \beta_1 m_t + (1-\beta_1) \nabla J$$

$$v_{t+1} = \beta_2 v_t + (1-\beta_2) (\nabla J)^2$$

$$\hat{m} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \qquad \hat{v} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon} \hat{m}$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\alpha = 10^{-3}$.

The bias correction terms $1/(1-\beta^{t+1})$ compensate for the fact that $m$ and $v$ are initialised at zero — without correction, early estimates are biased towards zero.

**Adam vs SGD+Momentum:** Adam converges faster and is more forgiving of hyperparameter choices. SGD+Momentum often achieves better final accuracy with careful tuning. Adam is the default for most tasks; SGD+Momentum is preferred when squeezing out maximum performance with a well-tuned schedule.

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(100, 10)

# ── All optimisers side by side ───────────────────────────────────────
opts = {
    'SGD':           optim.SGD(model.parameters(), lr=0.01),
    'SGD+Momentum':  optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'SGD+Nesterov':  optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True),
    'AdaGrad':       optim.Adagrad(model.parameters(), lr=0.01),
    'RMSProp':       optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9),
    'Adam':          optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)),
    'AdamW':         optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2),
}

x      = torch.randn(32, 100)
target = torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()

for name, opt in opts.items():
    opt.zero_grad()
    loss = loss_fn(model(x), target)
    loss.backward()
    opt.step()
    print(f'{name:15s}: loss = {loss.item():.4f}')
```

---

## 2  Learning Rate Scheduling

The learning rate $\alpha$ is the single most important hyperparameter. A good schedule:

1. **Starts moderate** (~0.1 for SGD, ~1e-3 for Adam) to make fast initial progress
2. **Decreases over time** to fine-tune into a sharp minimum
3. Optionally uses **warmup** to stabilise early training

### 2.1  Step Decay

Reduce the learning rate by a fixed factor every $k$ epochs:

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

Simple and widely used. Typical: $\gamma = 0.1$, $k = 30$ epochs.

### 2.2  Cosine Annealing

Smoothly decay the learning rate following a cosine curve from $\alpha_\text{max}$ to $\alpha_\text{min}$:

$$\alpha_t = \alpha_\text{min} + \frac{1}{2}(\alpha_\text{max} - \alpha_\text{min})\left(1 + \cos\!\frac{\pi t}{T}\right)$$

Smoother than step decay; no hyperparameter for when to step.

### 2.3  Warmup

For large batch training or Transformer models, start with a very small learning rate for the first few epochs (warmup), then increase to the target rate before decaying. This prevents instability caused by large gradient updates when the model is far from convergence.

```python
import torch.optim as optim
from torch.optim.lr_scheduler import (StepLR, CosineAnnealingLR,
                                       LinearLR, SequentialLR)

model = torch.nn.Linear(10, 1)
opt   = optim.SGD(model.parameters(), lr=0.1)

# Step decay: × 0.1 every 30 epochs
scheduler_step = StepLR(opt, step_size=30, gamma=0.1)

# Cosine annealing over 100 epochs
scheduler_cos = CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)

# Warmup (5 epochs) then cosine annealing
warmup    = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine    = CosineAnnealingLR(opt, T_max=95, eta_min=1e-6)
scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[5])

# Training loop skeleton
for epoch in range(100):
    # ... train ...
    scheduler.step()   # update LR after each epoch
    print(f'Epoch {epoch:3d}  LR = {opt.param_groups[0]["lr"]:.6f}')
```

---

## 3  Regularisation

### 3.1  Early Stopping

Monitor validation loss during training. Stop when it stops decreasing (or starts increasing). Save the checkpoint with the best validation loss. This is simple, free, and surprisingly effective.

### 3.2  Weight Decay (L2 Regularisation)

Add the L2 norm of the weights to the loss:

$$J_\text{reg} = J + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

The gradient update becomes: $\mathbf{w} \leftarrow \mathbf{w}(1 - \alpha\lambda) - \alpha \nabla J$. The term $(1-\alpha\lambda)$ shrinks weights towards zero at every step — hence "weight decay".

**AdamW** decouples weight decay from the adaptive learning rate (vanilla Adam's L2 regularisation is absorbed into the adaptive scaling and doesn't work correctly as true weight decay). Always use AdamW rather than Adam + L2 penalty.

### 3.3  Dropout

During training, independently zero each activation with probability $p$ (typically $p=0.5$ for FC layers, $p=0.1$–$0.3$ for conv layers):

$$\tilde{a}_i = \begin{cases} a_i / (1-p) & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}$$

The $1/(1-p)$ scaling (inverted dropout) keeps the expected value of activations the same regardless of $p$, so **no adjustment is needed at test time** — just disable dropout.

Dropout prevents co-adaptation: units cannot rely on specific other units being present, forcing each to learn more robust, independently useful features.

### 3.4  Data Augmentation

Artificially expand the training set by applying label-preserving transformations to existing images. For image classification:

- Random horizontal flips
- Random crops (with padding)
- Colour jitter (brightness, contrast, saturation, hue)
- Random rotation
- Mixup: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$
- CutOut / CutMix: remove or replace random rectangular patches

Data augmentation is usually the single most effective regulariser for image tasks because it directly increases the diversity of the training distribution.

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import v2

# ── Data augmentation pipeline ────────────────────────────────────────
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),        # CIFAR standard
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

# Test: NO augmentation (only normalisation)
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

# ── Dropout placement ─────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),   # FC: 50% dropout
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10),
)

# CRITICAL: switch modes between train and eval
x = torch.randn(4, 512)
model.train()
out_train = model(x)   # dropout active: different each forward pass

model.eval()
with torch.no_grad():
    out_eval1 = model(x)
    out_eval2 = model(x)
    
print(f'Eval outputs same: {torch.allclose(out_eval1, out_eval2)}')  # True
```

---

## 4  Hyperparameter Search

### 4.1  The Search Space

Always search learning rate on a **log scale**: $\alpha \in \{10^{-4}, 10^{-3}, \ldots, 10^{-1}\}$. The optimal LR can span several orders of magnitude; linear search would miss most of this range.

Similarly for regularisation coefficient $\lambda$: search over $\{10^{-5}, 10^{-4}, \ldots, 10^{-1}\}$.

### 4.2  Random vs Grid Search

For 2+ hyperparameters, **random search** outperforms grid search in most practical situations. The intuition: often only one or two hyperparameters matter most. Grid search wastes evaluations testing many values of unimportant hyperparameters, while random search independently samples each, giving more distinct values of the important ones.

### 4.3  Coarse-to-Fine Protocol

1. Run 20–30 random evaluations over a wide range, training for only 5–10 epochs each
2. Identify the region of hyperparameter space where validation loss is lowest
3. Run 10–20 more evaluations in that region, training longer
4. Identify the best configuration
5. Train with that configuration to convergence
6. Evaluate once on the **test set** — never again

### 4.4  Reading Loss Curves

| Symptom | Likely cause |
|---|---|
| Training loss does not decrease | LR too small, or bug in forward pass |
| Training loss explodes | LR too large |
| Training loss decreases, val loss does not | Overfitting — add regularisation or more data |
| Large gap between train and val from epoch 1 | Severe overfitting / dataset too small |
| Loss plateaus then drops | Learning rate decay kicked in (expected) |
| Spiky loss | Batch size too small, or bad data |
| Val loss lower than train loss | Dropout or augmentation active at train but not eval — check `model.eval()` |

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ── Coarse hyperparameter search ──────────────────────────────────────
import random, math

def random_lr():
    """Sample learning rate log-uniformly from [1e-4, 1e-1]."""
    return 10 ** random.uniform(-4, -1)

def random_wd():
    """Sample weight decay log-uniformly from [1e-5, 1e-2]."""
    return 10 ** random.uniform(-5, -2)

results = []
for trial in range(10):   # in practice: 20-50 trials
    lr = random_lr()
    wd = random_wd()

    model   = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 5))
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.randn(200, 50)
    y = torch.randint(0, 5, (200,))

    for epoch in range(10):   # short training run
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()

    with torch.no_grad():
        val_loss = loss_fn(model(X[:40]), y[:40]).item()
    results.append((val_loss, lr, wd))
    print(f'Trial {trial+1:2d}: lr={lr:.2e}  wd={wd:.2e}  val_loss={val_loss:.4f}')

best = min(results)
print(f'\nBest: lr={best[1]:.2e}  wd={best[2]:.2e}  val_loss={best[0]:.4f}')
```

---

## 5  Complete Training Recipe

Putting it all together: a checklist for training a CNN from scratch or fine-tuning.

```python
import torch, torch.nn as nn, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

# 1. DATA — augment train, only normalise val/test
MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
train_tf = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
test_tf  = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

# 2. MODEL — Kaiming init (PyTorch default), BatchNorm, Dropout
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c))
    def forward(self, x): return torch.relu(x + self.net(x))

model = nn.Sequential(
    nn.Conv2d(3,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    ResBlock(64), ResBlock(64),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(64, 10),
)

# 3. OPTIMISER — AdamW with weight decay
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 4. SCHEDULER — cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

# 5. TRAINING LOOP
loss_fn = nn.CrossEntropyLoss()

def train_epoch(loader):
    model.train()
    for X, y in loader:
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clip
        opt.step()
    scheduler.step()

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total

# 6. EVAL on val each epoch; save best; test ONCE at the end
best_val, best_state = 0, None
for epoch in range(5):   # 100 in practice
    train_epoch([])      # pass real dataloader here
    # val_acc = evaluate(val_loader)
    # if val_acc > best_val: best_val = val_acc; best_state = model.state_dict()
```

---

## 6  Summary

| Technique | What it fixes | Key parameter |
|---|---|---|
| SGD + Momentum | Slow convergence, oscillation | $\mu = 0.9$ |
| Adam | Sensitivity to LR tuning | $\beta_1=0.9, \beta_2=0.999$ |
| AdamW | Adam's incorrect L2 regularisation | `weight_decay` |
| LR warmup | Instability in early training | Warmup epochs |
| Cosine annealing | Manual LR schedule tuning | $T_\text{max}$ |
| Early stopping | Overfitting | Patience |
| Dropout | Co-adaptation of neurons | $p = 0.5$ (FC), $0.1$–$0.3$ (Conv) |
| Data augmentation | Small/non-diverse training set | Task-specific |
| Weight decay | Large weights (overfitting) | $\lambda \approx 10^{-4}$ to $10^{-2}$ |
| Gradient clipping | Exploding gradients (RNNs, deep nets) | Max norm $\approx 1.0$ |
| Random LR search | Efficient hyperparameter search | Log-uniform sampling |

## References

- Kingma, D. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR.
- Smith, L. (2017). Cyclical Learning Rates for Training Neural Networks. WACV.
