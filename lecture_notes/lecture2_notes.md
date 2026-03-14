# Lecture 2
# Machine Learning Fundamentals

*Deep Learning for Visual Recognition · Aarhus University*

These notes cover the core mathematical ideas of this lecture — loss functions, gradient descent, logistic regression, regularisation, softmax, and nearest neighbours — alongside PyTorch code that maps each concept directly to practice.

---

## 1  The Learning Principle

Every supervised machine learning algorithm is built around the same simple idea: given a dataset of input–output pairs $\{(\mathbf{x}^{(i)}, y^{(i)})\}$, find a function $h(\mathbf{x})$ — called the model or hypothesis — such that $h(\mathbf{x}^{(i)}) \approx y^{(i)}$ across the training set, and such that $h$ generalises to unseen data. The three moving parts are:

- **Model**: the family of functions $h_\mathbf{w}(\mathbf{x})$ parameterised by weights $\mathbf{w}$.
- **Loss function**: a scalar $J(\mathbf{w})$ that measures how badly the model's predictions differ from the true labels.
- **Optimiser**: an algorithm that adjusts $\mathbf{w}$ to minimise $J(\mathbf{w})$.

This lecture introduces the simplest instantiation of each: a linear model, the L2 or cross-entropy loss, and gradient descent. Everything in later lectures — deep CNNs, transformers, diffusion models — is built on exactly these three components.

---

## 2  Linear Regression

### 2.1  The Model

Linear regression addresses the problem of predicting a continuous scalar output $y \in \mathbb{R}$ from a feature vector $\mathbf{x} \in \mathbb{R}^m$. The model is a linear function of the weights:

$$h_\mathbf{w}(\mathbf{x}) = w_1 x_1 + w_2 x_2 + \cdots + w_m x_m = \mathbf{w}^T \mathbf{x}$$

The model is linear in the weights $\mathbf{w}$, even if the inputs $\mathbf{x}$ are non-linear (e.g. polynomial features). This distinction matters: gradient descent works cleanly on linear-in-weights models because the loss landscape is convex.

### 2.2  The Loss Function (L2 / MSE)

We need a way to score how well a given choice of $\mathbf{w}$ fits the training data. The standard choice for regression is the L2 loss (also called mean squared error, MSE):

$$J(\mathbf{w}) = \frac{1}{2} \sum_i \left( h_\mathbf{w}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2} \sum_i \left( \mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)} \right)^2$$

The factor of $\frac{1}{2}$ is just for mathematical convenience: it cancels with the exponent 2 when we differentiate, giving a cleaner gradient formula. Minimising $J(\mathbf{w})$ is our goal.

### 2.3  Gradient Descent

We cannot usually find the minimum of $J(\mathbf{w})$ in closed form for large models, so we use an iterative procedure: gradient descent. Imagine standing on a hilly landscape (the loss surface) and wanting to reach the lowest point. At each step, you look at which direction slopes steepest downhill and take a small step in that direction:

$$\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \cdot \nabla J(\mathbf{w}_k)$$

where $\alpha$ is the learning rate (step size) and $\nabla J(\mathbf{w}_k)$ is the gradient — a vector pointing in the direction of steepest ascent. Subtracting it moves us downhill. For the L2 loss, the gradient with respect to weight $w_j$ is:

$$\frac{\partial J}{\partial w_j} = \sum_i x_j^{(i)} \cdot \left( \mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)} \right)$$

This is the prediction error $(\mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)})$ weighted by the feature value $x_j^{(i)}$. Intuitively, if a feature $x_j$ is large and the model made a big error, $w_j$ gets a large corrective update.

> **Key insight: the chain rule.** The gradient $\partial J / \partial w_j$ is derived using the chain rule of differentiation. Let $p = \mathbf{w}^T\mathbf{x} - y$ (the error) and $q = \frac{1}{2}p^2$ (the squared error). Then $\partial q / \partial w = (\partial p / \partial w)(\partial q / \partial p) = x \cdot p = x(\mathbf{w}^T\mathbf{x} - y)$. This chain rule idea scales up directly to backpropagation in deep networks.

```python
import torch
import torch.nn as nn

# ── Linear regression from scratch ────────────────────────────────────

# Toy dataset: y = 2x + 1  plus some noise
torch.manual_seed(0)
X = torch.randn(100, 1)                    # 100 examples, 1 feature
y = 2 * X + 1 + 0.2 * torch.randn(100, 1) # true relationship + noise

# Model: a single linear layer (no bias separately; nn.Linear includes it)
model = nn.Linear(in_features=1, out_features=1)

# Loss and optimiser
loss_fn   = nn.MSELoss()                          # L2 / mean squared error
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(200):
    optimiser.zero_grad()           # 1. clear gradients from last step
    y_pred = model(X)               # 2. forward pass: compute predictions
    loss   = loss_fn(y_pred, y)     # 3. compute scalar loss
    loss.backward()                 # 4. backward pass: compute gradients
    optimiser.step()                # 5. update weights

    if epoch % 50 == 0:
        print(f'Epoch {epoch:3d}  Loss: {loss.item():.4f}')

# Inspect learned weights — should be close to w=2, b=1
w, b = model.weight.item(), model.bias.item()
print(f'Learned: y = {w:.3f}·x + {b:.3f}')   # ≈ y = 2.0·x + 1.0
```

*Code 1 – Linear regression in PyTorch. The five-line training loop (zero_grad → forward → loss → backward → step) is the universal PyTorch training pattern. `nn.MSELoss()` implements the L2 loss.*

### 2.4  The Learning Rate

The learning rate $\alpha$ is one of the most important hyperparameters in machine learning. Setting it incorrectly leads to two failure modes:

- **Too large**: Updates overshoot the minimum. The loss oscillates or diverges. Symptom: loss goes up and down erratically instead of decreasing.
- **Too small**: Updates are tiny. Training converges correctly but very slowly. Symptom: smooth but painfully slow decrease in loss.

It is generally good practice to start with a moderate learning rate (e.g. `1e-3`) and decay it over training so that early progress is fast and later fine-tuning is precise. We revisit this in Lecture 6.

### 2.5  Overfitting and Underfitting

These concepts are most easily illustrated with polynomial regression, where the degree of the polynomial is a hyperparameter controlling model capacity:

- **Underfitting (too low capacity)**: A degree-1 polynomial fit to data generated by a cubic will miss the curvature — the training loss is high.
- **Appropriate capacity**: The right polynomial degree captures the underlying pattern and generalises to new points.
- **Overfitting (too high capacity)**: A degree-15 polynomial can pass through every training point exactly, achieving zero training loss, but will wildly mispredict new data — it has memorised noise.

```python
import torch
import torch.nn as nn

# Polynomial regression via feature engineering
# The model is still LINEAR IN THE WEIGHTS — just the features are non-linear.

def poly_features(x, degree):
    """Expand scalar x into [x^0, x^1, ..., x^degree]."""
    return torch.cat([x ** d for d in range(degree + 1)], dim=1)

# True function: y = sin(x) + noise
torch.manual_seed(1)
x_train = torch.linspace(-3, 3, 20).unsqueeze(1)
y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)

for degree in [1, 4, 15]:
    X = poly_features(x_train, degree)     # (20, degree+1)
    model = nn.Linear(degree + 1, 1, bias=False)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2000):
        opt.zero_grad()
        loss = nn.MSELoss()(model(X), y_train)
        loss.backward()
        opt.step()
    print(f'Degree {degree:2d}  train loss: {loss.item():.5f}')

# Degree  1  → high loss (underfitting)
# Degree  4  → low loss (good fit)
# Degree 15  → near-zero loss (overfitting — memorised noise)
```

*Code 2 – Polynomial regression illustrating underfitting and overfitting. The model is still linear in the weights; only the input features are non-linear. A degree-15 fit will reach near-zero training loss but will perform poorly on new data.*

---

## 3  Hyperparameters, Train/Validation/Test Splits, and Cross-Validation

### 3.1  What Are Hyperparameters?

A hyperparameter is any setting that is chosen before training begins and is not updated by gradient descent. Contrast this with the model's parameters (weights), which are learned from data. Examples include:

- The learning rate $\alpha$.
- The degree of a polynomial (model capacity / architecture choice).
- The regularisation strength $\lambda$ (introduced in Section 4).
- The number of training epochs.

Choosing good hyperparameters is crucial — and the correct way to do it requires a careful data split strategy.

### 3.2  Train, Validation, and Test Sets

Using the test set to choose hyperparameters is a form of data leakage: the test set will no longer be a trustworthy estimate of real-world performance, because it has influenced the model design. The correct procedure is:

- **Training set**: used to compute gradients and update model weights.
- **Validation set**: used to evaluate the model during hyperparameter search. No gradients are computed; this set is only used to score different hyperparameter choices.
- **Test set**: touched exactly once, after all design decisions are final, to produce the reported performance number.

> **Why keep the test set secret?** Every time you look at test performance and make a decision based on it, you are implicitly fitting to the test set. With enough such decisions, you will overfit the test set and your reported performance will be optimistic. In deep learning this is a real and common problem.

### 3.3  Cross-Validation

When the dataset is small, a single validation split may be noisy — by bad luck, the validation set might be unrepresentative. Cross-validation addresses this by rotating which fold acts as validation:

- Split the (non-test) data into $k$ folds (e.g. $k = 5$).
- For each fold: train on the remaining $k-1$ folds, evaluate on this fold.
- Average the $k$ validation scores. This is the cross-validation estimate of performance.

Cross-validation is less common in deep learning (because training is expensive), but it is the right tool for small datasets or when comparing hyperparameter settings rigorously.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Standard train/val/test split ─────────────────────────────────────
torch.manual_seed(42)
N = 1000
X_all = torch.randn(N, 10)
y_all = torch.randint(0, 2, (N,))

# 70 / 15 / 15 split
n_train, n_val = int(0.7 * N), int(0.15 * N)
n_test  = N - n_train - n_val
dataset = TensorDataset(X_all, y_all)
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
test_loader  = DataLoader(test_ds,  batch_size=64)

print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

# ── Validation loop (no gradient computation) ─────────────────────────
def evaluate(model, loader, loss_fn):
    model.eval()                    # disables dropout, batch-norm update
    total_loss, correct = 0, 0
    with torch.no_grad():           # no gradient tracking needed
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            total_loss += loss_fn(logits, y_batch).item()
            correct    += (logits.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
```

*Code 3 – Splitting a dataset into train/validation/test in PyTorch. The validation loop uses `torch.no_grad()` to skip gradient computation and `model.eval()` to disable training-time behaviour (e.g. dropout).*

---

## 4  Logistic Regression

### 4.1  From Regression to Classification

Linear regression predicts a continuous value. When we instead want to assign one of two discrete class labels (e.g. '0' vs '1', 'dog' vs 'cat'), we need a classification algorithm. Logistic regression is the simplest such algorithm.

A naive approach — threshold the linear model: predict 1 if $\mathbf{w}^T\mathbf{x} > 0$, else 0 — fails with gradient descent because the threshold function has zero gradient almost everywhere. Instead, we use a smooth approximation that outputs a probability:

$$h_\mathbf{w}(\mathbf{x}) = P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{w}^T\mathbf{x})}$$

The sigmoid function $\sigma(z)$ squashes any real number into $(0, 1)$, making it interpretable as a probability. The predicted class label is then: predict 1 if $h_\mathbf{w}(\mathbf{x}) > 0.5$, else 0. Since $\sigma(z) > 0.5$ if and only if $z > 0$, this is equivalent to the sign of $\mathbf{w}^T\mathbf{x}$.

> **Geometric intuition.** The weight vector $\mathbf{w}$ defines a linear decision boundary: the hyperplane $\mathbf{w}^T\mathbf{x} = 0$. Points on one side are classified as class 1; points on the other side as class 0. The sigmoid converts the signed distance from this boundary into a probability.

### 4.2  The Cross-Entropy Loss

We could train logistic regression with MSE loss, but it is a poor choice: the loss landscape becomes non-convex and training is slow. The right loss for binary classification is the binary cross-entropy (log loss), derived from maximum likelihood estimation:

$$J(\mathbf{w}) = -\sum_i \left[ y^{(i)} \log h_\mathbf{w}(\mathbf{x}^{(i)}) + (1 - y^{(i)}) \log(1 - h_\mathbf{w}(\mathbf{x}^{(i)})) \right]$$

How to read this: for each training example, only one of the two log terms is active (the other is multiplied by zero). When $y = 1$, the loss is $-\log(\text{predicted probability of class 1})$: small loss if we predicted close to 1, large loss if we predicted close to 0. The negative log of a number in $(0, 1)$ is always positive and blows up as the number approaches 0 — exactly the behaviour we want.

The gradient of $J(\mathbf{w})$ with respect to $w_j$ turns out to have the same elegant form as for linear regression:

$$\frac{\partial J}{\partial w_j} = \sum_i x_j^{(i)} \cdot \left( h_\mathbf{w}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$

The only difference from the linear regression gradient is that $h_\mathbf{w}(\mathbf{x})$ is now the sigmoid of the linear prediction rather than the linear prediction itself.

```python
import torch
import torch.nn as nn

# ── Binary logistic regression on synthetic data ──────────────────────
torch.manual_seed(0)
N = 200
# Class 0: centred at (-1, -1);  Class 1: centred at (1, 1)
X0 = torch.randn(N // 2, 2) - 1
X1 = torch.randn(N // 2, 2) + 1
X  = torch.cat([X0, X1])
y  = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)]).long()

# nn.Linear gives us w^T x + b
# nn.BCEWithLogitsLoss = sigmoid + binary cross-entropy, numerically stable
model     = nn.Linear(2, 1)               # 2 input features, 1 output logit
loss_fn   = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    optimiser.zero_grad()
    logits = model(X).squeeze()           # shape: (N,)  — raw scores
    loss   = loss_fn(logits, y.float())
    loss.backward()
    optimiser.step()

# Compute accuracy
with torch.no_grad():
    probs    = torch.sigmoid(model(X).squeeze())
    preds    = (probs > 0.5).long()
    accuracy = (preds == y).float().mean()
    print(f'Accuracy: {accuracy:.2%}')   # should be ~99%

# Inspect the learned decision boundary
w = model.weight.data.squeeze()   # shape: (2,)
b = model.bias.data.item()
print(f'w = {w.numpy()},  b = {b:.3f}')
# The decision boundary is the line:  w[0]*x1 + w[1]*x2 + b = 0
```

*Code 4 – Binary logistic regression. `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a single numerically stable operation, which is why we pass raw logits (not probabilities) to it.*

### 4.3  Logistic Regression as an Image Classifier

On image data such as MNIST, we flatten the 28×28 pixel grid into a 784-dimensional vector and apply logistic regression directly. The weight vector $\mathbf{w}$ then has the same shape as the input image and can be visualised as a 'template': the model classifies an image by measuring how similar it is (via inner product) to the learned template.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Load MNIST and flatten to vectors ─────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean & std
    transforms.Lambda(lambda x: x.view(-1)),      # flatten 28x28 → 784
])

train_ds = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512)

# ── Softmax classifier (logistic regression for 10 classes) ───────────
# nn.Linear maps each 784-dim image to 10 class scores
model     = nn.Linear(784, 10)              # 784*10 + 10 = 7,850 parameters
loss_fn   = nn.CrossEntropyLoss()           # softmax + cross-entropy
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    model.train()
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimiser.step()

    # Validation accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()
    print(f'Epoch {epoch+1}  test accuracy: {correct/len(test_ds):.2%}')
# Typically reaches ~92% in 5 epochs — not bad for a linear model!
```

*Code 5 – Logistic/softmax regression on MNIST. `nn.CrossEntropyLoss` expects raw logits (not probabilities) and internally applies log-softmax. A linear model on raw pixels achieves ~92% on MNIST — the ceiling of what a linear classifier can do.*

---

## 5  Regularisation: Weight Decay

### 5.1  Why Regularise?

When a model has more capacity than the data can support — for example, a degree-15 polynomial fitted to 20 noisy points — it overfits: training loss is very low but generalisation is poor. Regularisation adds a penalty term to the loss that discourages overly complex models:

$$J_\text{reg}(\mathbf{w}) = J(\mathbf{w}) + \lambda \cdot R(\mathbf{w})$$

where $R(\mathbf{w})$ is a regularisation term and $\lambda > 0$ is the regularisation strength (a hyperparameter). A larger $\lambda$ means stronger regularisation — more pressure to keep weights small — at the cost of potentially underfitting.

### 5.2  L2 Regularisation (Weight Decay)

L2 regularisation (also called weight decay) penalises the sum of squared weights:

$$R(\mathbf{w}) = \sum_j w_j^2 = \|\mathbf{w}\|^2$$

Adding this to the gradient descent update gives:

$$w_j \leftarrow w_j - \alpha \cdot \left(\frac{\partial J}{\partial w_j} + 2\lambda w_j\right) = (1 - 2\alpha\lambda) \cdot w_j - \alpha \cdot \frac{\partial J}{\partial w_j}$$

The factor $(1 - 2\alpha\lambda)$ shrinks the weight at every step — hence 'weight decay'. L2 regularisation has an analytical solution and tends to give dense solutions where all weights are small but non-zero.

### 5.3  L1 Regularisation

L1 regularisation penalises the sum of absolute weight values:

$$R(\mathbf{w}) = \sum_j |w_j| = \|\mathbf{w}\|_1$$

L1 regularisation has a fundamentally different effect: it tends to produce sparse solutions where many weights are exactly zero. This is a form of automatic feature selection — the model learns to ignore most inputs. In high-dimensional settings this can be very useful, but in deep learning L2 is far more common.

> **L1 vs L2 geometrically.** Imagine the set of all weight vectors that fit the training data exactly (a line in $w_1$–$w_2$ space). L2 picks the one with the smallest Euclidean distance from the origin (always gives a non-sparse, 'spread-out' solution). L1 picks the one with the smallest Manhattan distance — and this solution typically touches a corner of the L1 ball, where one weight is zero.

```python
import torch
import torch.nn as nn

# ── L2 regularisation in PyTorch ──────────────────────────────────────
# Option 1: Pass weight_decay to the optimiser (most common, cleanest)
model     = nn.Linear(784, 10)
optimiser = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4    # λ  — applied as L2 penalty on all parameters
)

# Option 2: Manual L2 penalty added to the loss (more transparent)
loss_fn = nn.CrossEntropyLoss()
lambda_ = 1e-4

X_batch = torch.randn(32, 784)
y_batch = torch.randint(0, 10, (32,))

logits    = model(X_batch)
data_loss = loss_fn(logits, y_batch)
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
total_loss = data_loss + lambda_ * l2_penalty
print(f'Data loss: {data_loss:.4f}  L2 penalty: {(lambda_*l2_penalty):.4f}')

# ── L1 regularisation (manual — no built-in optimiser shortcut) ───────
l1_penalty = sum(p.abs().sum() for p in model.parameters())
total_loss_l1 = data_loss + lambda_ * l1_penalty

# L1 will push some weights to exactly zero — useful for feature selection
```

*Code 6 – L2 and L1 regularisation in PyTorch. In practice, L2 is almost always applied via the `weight_decay` argument to the optimiser. L1 must be added manually to the loss, as PyTorch optimisers do not have a built-in L1 option.*

---

## 6  Softmax Regression (Multi-Class Classification)

### 6.1  From Binary to K Classes

Logistic regression handles two classes. For $K > 2$ classes we use softmax regression (also called multinomial logistic regression). Instead of a single weight vector $\mathbf{w}$, we learn a weight matrix $\mathbf{W}$ with one row $\mathbf{w}_k$ per class:

$$h_\mathbf{W}(\mathbf{x}) = \text{softmax}(\mathbf{W}\mathbf{x}) \quad \text{where} \quad \text{softmax}(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

Each class score $z_k = \mathbf{w}_k^T\mathbf{x}$ is the inner product of that class's weight vector with the input — a measure of how similar the input is to that class's template. The softmax function converts these $K$ raw scores (logits) into $K$ probabilities that sum to 1. We take the class with the highest probability as our prediction.

Two design questions answered by softmax:

- **Why $\exp(z)$ rather than $z$ directly?** Probabilities must be non-negative, and $\exp()$ guarantees this for any real-valued logit.
- **Why divide by $\sum \exp(z_j)$?** To normalise the outputs so they sum to 1 and form a valid probability distribution.

### 6.2  Loss Function: Cross-Entropy

The loss for softmax regression is the multi-class cross-entropy. Given the predicted probability distribution $h_\mathbf{W}(\mathbf{x}^{(i)})$ and the one-hot target vector for training example $i$ (all zeros except a 1 in the position of the true class $k$):

$$J(\mathbf{W}) = -\sum_i \sum_k \mathbf{1}[y^{(i)} = k] \cdot \log P(y = k \mid \mathbf{x}^{(i)})$$

Because of the indicator function $\mathbf{1}[\cdot]$, only the term corresponding to the true class is non-zero for each example. The loss thus reduces to: $-\log(\text{predicted probability of the correct class})$. This is large when the model assigns low probability to the right class, and small when it is confident and correct.

```python
import torch
import torch.nn as nn

# ── Softmax regression: the forward pass step by step ─────────────────
batch_size, n_features, n_classes = 4, 8, 3

W     = torch.randn(n_classes, n_features, requires_grad=True)
b     = torch.zeros(n_classes, requires_grad=True)
x     = torch.randn(batch_size, n_features)
y     = torch.tensor([0, 2, 1, 0])   # true class labels

# Step 1: Compute logits (class scores)
logits = x @ W.T + b               # shape: (batch, n_classes)

# Step 2: Softmax converts logits to probabilities
probs = torch.softmax(logits, dim=1)
print('Probabilities (should sum to 1 per row):')
print(probs.detach().round(decimals=3))
print('Row sums:', probs.sum(dim=1).detach())

# Step 3: Cross-entropy loss
# nn.CrossEntropyLoss = log_softmax + NLLLoss — takes raw logits, not probs
loss_fn = nn.CrossEntropyLoss()
loss    = loss_fn(logits, y)
print(f'\nCross-entropy loss: {loss.item():.4f}')

# Step 4: Backward pass
loss.backward()
print(f'Gradient of W: shape {W.grad.shape}')   # same as W

# ── Using nn.Linear (cleaner implementation) ──────────────────────────
model = nn.Sequential(
    nn.Linear(n_features, n_classes),   # W and b handled automatically
)
# nn.CrossEntropyLoss expects LOGITS (not softmax output)
loss2 = nn.CrossEntropyLoss()(model(x), y)
```

*Code 7 – Softmax regression broken down step by step. Important: `nn.CrossEntropyLoss` expects raw logits, not probabilities — it applies log-softmax internally for numerical stability. Never pass `torch.softmax(logits)` into `CrossEntropyLoss`.*

### 6.3  Linear Decision Boundaries and Their Limits

Each row $\mathbf{w}_k$ of the weight matrix $\mathbf{W}$ defines a linear decision boundary between class $k$ and the rest. For many real-world problems — images in particular — these linear boundaries are insufficient. Consider images of the digit '1' rotated 90° versus upright: they occupy completely different regions of pixel space, and no linear boundary cleanly separates them.

The solution is non-linear feature transformations — either hand-crafted (e.g. converting Cartesian to polar coordinates) or, far better, learned by a neural network. This is the direct motivation for moving from softmax regression to multi-layer neural networks (Lecture 3) and then convolutional networks (Lecture 4).

```python
import torch
import torch.nn as nn

# ── Visualising the limits of a linear classifier ─────────────────────
# XOR problem: NOT linearly separable
# Class 0: (0,0) and (1,1)   Class 1: (0,1) and (1,0)
X = torch.tensor([[0.,0.],[1.,1.],[0.,1.],[1.,0.]])
y = torch.tensor([0, 0, 1, 1])

# Linear model (logistic regression)
lin_model = nn.Linear(2, 2)
opt       = torch.optim.Adam(lin_model.parameters(), lr=0.1)
loss_fn   = nn.CrossEntropyLoss()

for _ in range(1000):
    opt.zero_grad()
    loss_fn(lin_model(X), y).backward()
    opt.step()

preds_lin = lin_model(X).argmax(1)
print('Linear model predictions:', preds_lin.tolist())  # will fail on XOR

# Non-linear model (one hidden layer with ReLU)
mlp = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))
opt2 = torch.optim.Adam(mlp.parameters(), lr=0.05)

for _ in range(2000):
    opt2.zero_grad()
    loss_fn(mlp(X), y).backward()
    opt2.step()

preds_mlp = mlp(X).argmax(1)
print('MLP predictions:', preds_mlp.tolist())   # correctly classifies XOR
```

*Code 8 – The XOR problem illustrates the fundamental limit of linear classifiers. No straight line can separate the two classes. A single hidden layer with ReLU solves it easily — motivating neural networks.*

---

## 7  K-Nearest Neighbours (K-NN)

### 7.1  The Algorithm

K-NN is the simplest classification algorithm imaginable: it requires no training at all. Given a new test image, it finds the $K$ training images most similar to it (by some distance metric) and assigns the majority class among those $K$ neighbours.

- $K = 1$: Assign the label of the single closest training example. Produces jagged, noisy decision boundaries.
- $K > 1$: Majority vote among $K$ neighbours. Smoother boundaries, more robust to individual noisy examples.

The 'white regions' visible in K-NN decision boundary plots correspond to ties in the majority vote — equally many neighbours from two or more classes.

### 7.2  Computational Complexity

The key practical disadvantage of K-NN:

- **Training**: $O(1)$ — just store all training examples.
- **Prediction**: $O(N)$ — must compute distance to every training example for each new query.

This is exactly backwards from what we want for deployment: cheap training is fine, but slow prediction is painful. For $N = 1{,}000{,}000$ training images, each prediction requires a million distance computations. Data structures like KD-trees can speed this up, but the fundamental scaling problem remains.

### 7.3  The Curse of Dimensionality

K-NN on raw pixels performs poorly for a deeper reason than just speed. CIFAR-10 images are $32 \times 32 \times 3 = 3072$-dimensional. In high-dimensional spaces, all data points become approximately equally distant from each other — the concept of 'nearest neighbour' breaks down because distances stop being informative. An image of a white cat and an image of a black dog may be closer in pixel space (same brightness distribution) than two images of the same cat in different lighting.

The solution previewed at the end of the lecture is to use a CNN to extract a compact, semantically meaningful representation (e.g. a 512-dimensional vector) before applying K-NN. In that learned space, semantic similarity and geometric proximity align.

```python
import torch
import torch.nn.functional as F

# ── K-NN implemented with PyTorch (vectorised, no loops) ──────────────
class KNN:
    def __init__(self, k=5, metric='l2'):
        self.k      = k
        self.metric = metric

    def fit(self, X_train, y_train):
        """Store training data — no computation here."""
        self.X_train = X_train   # shape: (N_train, D)
        self.y_train = y_train   # shape: (N_train,)

    def predict(self, X_test):
        """Find K nearest neighbours for each test point."""
        if self.metric == 'l2':
            # Efficient pairwise L2 distance using broadcasting
            dists = torch.cdist(X_test, self.X_train, p=2)  # (N_test, N_train)
        else:  # L1
            dists = torch.cdist(X_test, self.X_train, p=1)

        # For each test point, find the K smallest distances
        _, topk_idx = dists.topk(self.k, dim=1, largest=False)  # (N_test, K)

        # Majority vote among K neighbours
        neighbor_labels = self.y_train[topk_idx]   # (N_test, K)
        preds = torch.mode(neighbor_labels, dim=1).values
        return preds

# ── Toy demo ──────────────────────────────────────────────────────────
torch.manual_seed(0)
X_train = torch.randn(200, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()  # simple rule
X_test  = torch.randn(20, 2)
y_test  = (X_test[:, 0]  + X_test[:, 1]  > 0).long()

for k in [1, 5, 15]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    preds    = knn.predict(X_test)
    accuracy = (preds == y_test).float().mean()
    print(f'K={k:2d}  accuracy: {accuracy:.0%}')
```

*Code 9 – K-NN implemented in PyTorch using `torch.cdist` for efficient vectorised pairwise distance computation. Notice that `fit()` is trivial (just storing data), while `predict()` does all the work — $O(N_\text{train})$ per test point.*

---

## 8  K-Means Clustering

### 8.1  Unsupervised Learning

All algorithms so far have been supervised: we learn from labelled pairs $(\mathbf{x}, y)$. Unsupervised learning uses unlabelled data — only the inputs $\mathbf{x}$ — to discover structure. This is valuable when labels are expensive or unavailable, which is common in practice.

K-Means is the simplest and most widely used clustering algorithm. It partitions $N$ data points into $K$ clusters by iteratively assigning points to their nearest cluster centre (centroid) and updating the centroids.

### 8.2  The Algorithm

- Initialise $K$ cluster centroids $\mu_1, \ldots, \mu_K$ (e.g. random points from the dataset).
- **Assignment step**: assign each point $\mathbf{x}_i$ to the nearest centroid: $c_i = \arg\min_k \|\mathbf{x}_i - \mu_k\|^2$
- **Update step**: recompute each centroid as the mean of all points assigned to it: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$
- Repeat assignment and update steps until assignments stop changing (convergence).

K-Means minimises the within-cluster sum of squared distances. It is guaranteed to converge but only to a local minimum, so it is common practice to run it multiple times with different initialisations and keep the best result.

```python
import torch

def kmeans(X, K, n_iters=100, seed=0):
    """
    K-Means clustering.
    Args:
        X: (N, D) tensor of data points
        K: number of clusters
    Returns:
        centroids: (K, D) final cluster centres
        assignments: (N,) cluster index for each point
    """
    torch.manual_seed(seed)
    N, D = X.shape

    # Initialise: pick K random data points as starting centroids
    idx       = torch.randperm(N)[:K]
    centroids = X[idx].clone()   # (K, D)

    for iteration in range(n_iters):
        # ── Assignment step ────────────────────────────────────────────
        # dists[i, k] = squared distance from point i to centroid k
        dists       = torch.cdist(X, centroids, p=2)  # (N, K)
        assignments = dists.argmin(dim=1)             # (N,) — closest centroid

        # ── Update step ────────────────────────────────────────────────
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = (assignments == k)
            if mask.any():                        # avoid empty clusters
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                new_centroids[k] = centroids[k]  # keep old centroid

        # Check convergence
        if (new_centroids - centroids).abs().max() < 1e-6:
            print(f'Converged at iteration {iteration}')
            break
        centroids = new_centroids

    return centroids, assignments

# ── Demo: cluster three Gaussian blobs ────────────────────────────────
torch.manual_seed(42)
blobs = torch.cat([
    torch.randn(50, 2) + torch.tensor([-3.,  0.]),
    torch.randn(50, 2) + torch.tensor([ 3.,  0.]),
    torch.randn(50, 2) + torch.tensor([ 0.,  3.]),
])

centroids, assignments = kmeans(blobs, K=3)
for k in range(3):
    count = (assignments == k).sum().item()
    print(f'Cluster {k}: {count} points, centroid ≈ {centroids[k].tolist()}')
```

*Code 10 – K-Means from scratch in PyTorch. The assignment step (argmin of pairwise distances) and update step (mean of assigned points) directly implement the algorithm from the slides. In practice, `sklearn.cluster.KMeans` is more efficient and robust, but this implementation makes the mechanics transparent.*

---

## 9  Summary

This lecture established the three core building blocks that every algorithm in this course is built on:

| Concept | What it does | PyTorch |
|---|---|---|
| Linear model | $h_\mathbf{w}(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$ — basis of regression and classification | `nn.Linear(m, 1)` |
| L2 loss (MSE) | Measures regression error; convex, easy to optimise | `nn.MSELoss()` |
| Cross-entropy loss | Measures classification error; derived from max-likelihood | `nn.CrossEntropyLoss()` |
| Gradient descent | Iteratively update $\mathbf{w} \leftarrow \mathbf{w} - \alpha\nabla J(\mathbf{w})$ to minimise the loss | `torch.optim.SGD(lr=α)` |
| Sigmoid | Maps any real number to $(0,1)$; used for binary probs | `torch.sigmoid(z)` |
| Softmax | Maps $K$ scores to a probability distribution over $K$ classes | `torch.softmax(z, dim=1)` |
| L2 regularisation | Penalises large weights to prevent overfitting | `weight_decay=` in optim |
| L1 regularisation | Penalises absolute weight values; induces sparsity | manual: `p.abs().sum()` |
| K-NN | Non-parametric; predict by majority vote of $K$ neighbours | `torch.cdist(Xtest, Xtrain)` |
| K-Means | Unsupervised clustering; alternates assign and update | sklearn or manual |
| Train/Val/Test | Proper evaluation protocol to avoid data leakage | `random_split(dataset, ...)` |

The most important concept to carry forward is the three-step recipe: define a model, choose a loss, run gradient descent. In Lecture 3 we stack multiple linear layers with non-linear activations to build neural networks that can represent arbitrarily complex functions — but the recipe stays exactly the same.

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 3, 4, 5.
- Andrew Ng, Unsupervised Feature Learning and Deep Learning Tutorial: http://deeplearning.stanford.edu/tutorial/
- Stanford CS229 lecture notes (Ng): http://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf
- PyTorch documentation: https://pytorch.org/docs/stable/nn.html
- Neural network playground (interactive): https://playground.tensorflow.org/