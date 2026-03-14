# Lecture 2 — Machine Learning Fundamentals

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes build machine learning from first principles, starting from linear regression as a vehicle for understanding loss functions and optimisation, through logistic and softmax regression, and finishing with k-nearest neighbours and k-means clustering. Every concept is connected to how it scales up to deep networks in later lectures.

---

## 1  The Learning Principle

The fundamental goal of supervised machine learning is to find a function $h(\mathbf{x})$ that maps inputs $\mathbf{x}$ to outputs $y$ such that $y^{(i)} \approx h(\mathbf{x}^{(i)})$ for every training example $i$, and — crucially — that also **generalises** to unseen data.

A model that memorises training examples perfectly but fails on new inputs has overfit; a model that is too simple to capture the structure in the data has underfit. The tension between these two failure modes runs through every topic in this lecture.

---

## 2  Linear Regression

### 2.1  Model

Given a dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}$ where $\mathbf{x}^{(i)} \in \mathbb{R}^m$ and $y^{(i)} \in \mathbb{R}$, a linear model predicts:

$$h_\mathbf{w}(\mathbf{x}) = \sum_{j=1}^{m} w_j x_j = \mathbf{w}^T \mathbf{x}$$

The vector $\mathbf{w} \in \mathbb{R}^m$ contains all learnable parameters. The family of all possible linear functions parameterised by $\mathbf{w}$ is called the *hypothesis class*.

### 2.2  Loss Function

We want $\mathbf{w}$ such that the predictions are as close as possible to the targets. The **mean squared error** (L2 loss) measures this:

$$J(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{n} \left( h_\mathbf{w}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2} \sum_{i=1}^{n} \left( \mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)} \right)^2$$

The factor of $\frac{1}{2}$ is a convenience that cancels the exponent when differentiating.

### 2.3  Gradient Descent

We minimise $J(\mathbf{w})$ iteratively. Starting from an initial guess $\mathbf{w}_0$ (zeros or random), we repeatedly step in the direction opposite to the gradient:

$$\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \nabla J(\mathbf{w}_k)$$

where $\alpha > 0$ is the **learning rate** (step size). For small enough $\alpha$, each step is guaranteed to decrease $J$.

The gradient of $J$ with respect to the $j$-th weight is derived via the chain rule:

$$\frac{\partial J(\mathbf{w})}{\partial w_j} = \sum_{i=1}^{n} x_j^{(i)} \left( \mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)} \right)$$

This says: for each training example, the residual $(\mathbf{w}^T\mathbf{x}^{(i)} - y^{(i)})$ is scaled by the $j$-th feature value $x_j^{(i)}$ and summed. Features that correlate with the prediction error drive the largest weight updates.

**Chain rule reminder.** If $p = g(\mathbf{w})$ and $q = f(p)$, then $\frac{dq}{dw} = \frac{dp}{dw} \cdot \frac{dq}{dp}$. For linear regression:

$$p = \mathbf{w}^T\mathbf{x} - y, \quad q = \tfrac{1}{2}p^2, \quad \Rightarrow \quad \frac{dq}{dw_j} = x_j \cdot p = x_j(\mathbf{w}^T\mathbf{x} - y)$$

### 2.4  Hyperparameters vs Parameters

- **Parameters** (e.g. $\mathbf{w}$): learned during training by gradient descent.
- **Hyperparameters** (e.g. learning rate $\alpha$, polynomial degree): set *before* training begins, not learned from data.

The correct procedure for selecting hyperparameters is to split data into **train / validation / test** sets:

1. Train the model on the training set.
2. Evaluate different hyperparameter choices on the **validation set**.
3. Report final performance on the **test set** — a set that is never touched during development.

Using the test set for hyperparameter selection is a common mistake; it leaks test information into the model development process and gives an optimistic performance estimate.

```python
import torch
import torch.nn as nn

# ── Linear regression from scratch ──────────────────────────────────
torch.manual_seed(0)

# Generate synthetic data: y = 2x + 1 + noise
n = 100
X = torch.randn(n, 1)
y = 2 * X + 1 + 0.3 * torch.randn(n, 1)

# Split 80/20 train/validation
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# Simple linear model: one weight + bias
model = nn.Linear(1, 1)
opt   = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Validation loss
model.eval()
with torch.no_grad():
    val_loss = loss_fn(model(X_val), y_val)
    w, b = model.weight.item(), model.bias.item()

print(f'Learned: y = {w:.3f}x + {b:.3f}')  # should be close to 2x+1
print(f'Val MSE: {val_loss:.4f}')
```

---

## 3  Logistic Regression

### 3.1  From Regression to Classification

When the target $y^{(i)} \in \{0, 1\}$ is a class label rather than a continuous value, we need a classification model. The naive approach of thresholding a linear model is not differentiable; we need a smooth output.

### 3.2  The Sigmoid Function

Logistic regression uses the **sigmoid** (logistic) function to squash the linear activation into $(0, 1)$, making it interpretable as a probability:

$$h_\mathbf{w}(\mathbf{x}) = P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

The complementary probability is:

$$P(y = 0 \mid \mathbf{x}) = 1 - h_\mathbf{w}(\mathbf{x})$$

The decision rule is: predict class 1 if $h_\mathbf{w}(\mathbf{x}) > 0.5$, which is equivalent to $\mathbf{w}^T\mathbf{x} > 0$.

### 3.3  Binary Cross-Entropy Loss

The loss function for logistic regression is the **binary cross-entropy**:

$$J(\mathbf{w}) = -\sum_{i=1}^{n} \left[ y^{(i)} \log h_\mathbf{w}(\mathbf{x}^{(i)}) + (1 - y^{(i)}) \log \left(1 - h_\mathbf{w}(\mathbf{x}^{(i)})\right) \right]$$

For each example, exactly one of the two log terms is non-zero:

- When $y^{(i)} = 1$: loss $= -\log h_\mathbf{w}(\mathbf{x}^{(i)})$, which is minimised when $h_\mathbf{w} \to 1$.
- When $y^{(i)} = 0$: loss $= -\log(1 - h_\mathbf{w}(\mathbf{x}^{(i)}))$, which is minimised when $h_\mathbf{w} \to 0$.

The negative sign makes the loss positive (since $\log x < 0$ for $0 < x < 1$) and ensures it has a minimum of zero when predictions are perfect.

**Maximum likelihood derivation.** The loss can be derived by observing that the probability of a binary label $y$ given input $\mathbf{x}$ can be written compactly as:

$$P(y \mid \mathbf{x}) = h_\mathbf{w}(\mathbf{x})^y \left(1 - h_\mathbf{w}(\mathbf{x})\right)^{1-y}$$

The likelihood of the entire training set (assuming independence) is:

$$L(\mathbf{w}) = \prod_{i=1}^{n} h_\mathbf{w}(\mathbf{x}^{(i)})^{y^{(i)}} \left(1 - h_\mathbf{w}(\mathbf{x}^{(i)})\right)^{1-y^{(i)}}$$

Minimising the cross-entropy loss $J(\mathbf{w}) = -\log L(\mathbf{w})$ is therefore equivalent to **maximum likelihood estimation** — finding the weights that make the observed labels most probable.

### 3.4  Gradient

The gradient of the cross-entropy loss for logistic regression has the same form as for linear regression:

$$\frac{\partial J(\mathbf{w})}{\partial w_j} = \sum_{i=1}^{n} x_j^{(i)} \left( h_\mathbf{w}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$

The only difference is that $h_\mathbf{w}(\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$ rather than $\mathbf{w}^T\mathbf{x}$ directly.

```python
import torch
import torch.nn as nn

# ── Logistic regression on MNIST (digits 0 vs 1) ─────────────────────
# (Simulated data — replace with real MNIST in practice)
torch.manual_seed(0)
n = 200
X = torch.randn(n, 784)            # 28×28 images flattened
y = (torch.randn(n) > 0).float()  # binary labels

model   = nn.Linear(784, 1)        # one output logit
loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + binary cross-entropy
opt     = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    logits = model(X).squeeze(1)
    loss   = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Predictions
with torch.no_grad():
    probs = torch.sigmoid(model(X).squeeze(1))
    preds = (probs > 0.5).float()
    acc   = (preds == y).float().mean()
    print(f'Training accuracy: {acc:.2%}')

# Intuition: the learned weight vector w, when reshaped to (28, 28),
# looks like a template for the class it detects.
# Positive pixels contribute to P(y=1); negative pixels suppress it.
print(f'Weight vector shape: {model.weight.shape}')  # (1, 784)
```

---

## 4  Regularisation: Weight Decay

### 4.1  The Overfitting Problem

A model with too many parameters relative to the number of training examples can fit the training data almost perfectly — including the noise — while generalising poorly. This is overfitting. Regularisation adds a penalty to the loss that discourages large weights, implicitly constraining the model's effective capacity.

### 4.2  Regularised Loss

The regularised loss adds a penalty term $R(\mathbf{w})$ scaled by a regularisation coefficient $\lambda$:

$$J_{\text{reg}}(\mathbf{w}) = J(\mathbf{w}) + \lambda R(\mathbf{w})$$

Two common choices for $R(\mathbf{w})$:

$$\text{L1 norm:} \quad R(\mathbf{w}) = \sum_i |w_i|$$

$$\text{L2 norm:} \quad R(\mathbf{w}) = \sum_i w_i^2$$

L2 regularisation is also called **weight decay**, because the gradient update for L2 shrinks every weight by a multiplicative factor at each step:

$$\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \left( \nabla J(\mathbf{w}_k) + 2\lambda \mathbf{w}_k \right) = (1 - 2\alpha\lambda)\mathbf{w}_k - \alpha \nabla J(\mathbf{w}_k)$$

### 4.3  L1 vs L2: Geometric Intuition

The key difference between L1 and L2 can be understood geometrically. Consider finding a solution on the constraint surface $\mathbf{w}^T\mathbf{x} = 1$ with $\mathbf{x} = (1,1,1,1)^T$ that minimises the regularisation norm:

- **L1 prefers sparse solutions**: the L1 unit ball has corners on the coordinate axes. The constrained optimum tends to land at a corner, where most weights are zero. L1 is a form of implicit feature selection.
- **L2 prefers spread-out solutions**: the L2 unit ball is smooth (a sphere). The constrained optimum lands where the sphere just touches the constraint surface, producing a unique, non-sparse solution. L2 distributes weight mass across all features.

For the example above, L1 gives one of $\{(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)\}$; L2 gives the unique solution $(0.25, 0.25, 0.25, 0.25)$.

---

## 5  Softmax Regression

### 5.1  Multi-Class Extension

Logistic regression handles two classes. **Softmax regression** (multinomial logistic regression) extends this to $K$ classes. The model learns a separate weight vector $\mathbf{w}_k$ for each class and outputs a probability distribution over all $K$ classes:

$$h_\mathbf{W}(\mathbf{x}) = \begin{pmatrix} P(y=1 \mid \mathbf{x}) \\ P(y=2 \mid \mathbf{x}) \\ \vdots \\ P(y=K \mid \mathbf{x}) \end{pmatrix} = \frac{1}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T\mathbf{x})} \begin{pmatrix} \exp(\mathbf{w}_1^T\mathbf{x}) \\ \exp(\mathbf{w}_2^T\mathbf{x}) \\ \vdots \\ \exp(\mathbf{w}_K^T\mathbf{x}) \end{pmatrix}$$

The **exp** ensures all values are positive; the denominator normalises them to sum to 1.

The weight matrix $\mathbf{W} = [\mathbf{w}_1 \mid \mathbf{w}_2 \mid \cdots \mid \mathbf{w}_K] \in \mathbb{R}^{m \times K}$ is the model's only learnable parameter.

### 5.2  Cross-Entropy Loss

The multi-class cross-entropy loss uses the indicator function $\mathbf{1}[\cdot]$ (equals 1 when true, 0 otherwise):

$$J(\mathbf{W}) = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y^{(i)} = k] \log \frac{\exp(\mathbf{w}_k^T \mathbf{x}^{(i)})}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x}^{(i)})}$$

For each training example, only the term corresponding to the true class $k = y^{(i)}$ is non-zero. The loss penalises the model for assigning low probability to the correct class.

### 5.3  Gradient

The gradient with respect to $\mathbf{w}_k$ is:

$$\nabla_{\mathbf{w}_k} J(\mathbf{W}) = -\sum_{i=1}^{n} \mathbf{x}^{(i)} \left( \mathbf{1}[y^{(i)} = k] - P(y^{(i)} = k \mid \mathbf{x}^{(i)}) \right)$$

This has a clean interpretation: for each example, the gradient is $\mathbf{x}^{(i)}$ scaled by the error in the model's probability estimate for class $k$. If the model is confident and correct, the error $(\mathbf{1}[\cdot] - P(\cdot))$ is near zero and no update occurs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Softmax regression on CIFAR-10 ───────────────────────────────────
# Input: 32×32×3 images = 3072-dimensional vectors
# Output: 10 class probabilities

torch.manual_seed(0)
n = 500
X = torch.randn(n, 3072)              # flattened CIFAR images
y = torch.randint(0, 10, (n,))       # class labels 0..9

model   = nn.Linear(3072, 10)         # W matrix + bias
loss_fn = nn.CrossEntropyLoss()       # softmax + cross-entropy
opt     = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(200):
    logits = model(X)                  # (n, 10) — raw class scores
    loss   = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    preds = model(X).argmax(1)
    acc   = (preds == y).float().mean()
    print(f'Training accuracy: {acc:.2%}')

# Each row of model.weight is a class template w_k.
# When reshaped to (3, 32, 32), it shows what colour/texture
# pattern the model associates most strongly with each class.
print(f'Weight matrix shape: {model.weight.shape}')  # (10, 3072)
```

---

## 6  K-Nearest Neighbours

### 6.1  The Algorithm

K-Nearest Neighbours (k-NN) is a non-parametric classifier: it stores all training examples and classifies a test point by majority vote among its $k$ closest training examples. Distance is typically measured with the L1 or L2 norm:

$$L_1 = \sum_p |I_1^p - I_2^p|, \qquad L_2 = \sqrt{\sum_p (I_1^p - I_2^p)^2}$$

**Training complexity:** $O(1)$ — just store the data.  
**Prediction complexity:** $O(N)$ — compare the test point against all $N$ training examples.

This is the opposite of what we want: we prefer fast prediction even at the cost of slow training.

### 6.2  Why k-NN Fails on Raw Pixels

Applying k-NN directly to pixel intensities performs poorly because pixel-level distances do not correspond to perceptual or semantic similarity. Two images of the same cat in different lighting conditions can be far apart in pixel space; a cat image and a car image shifted so their backgrounds match can be very close. The **curse of dimensionality** compounds the problem: CIFAR-10 images live in a $3072$-dimensional space where all points appear sparse and the notion of a meaningful "nearest neighbour" breaks down.

The fix is to use a better feature space — CNN features, for instance — where semantically similar images are geometrically close.

```python
import torch
import torch.nn.functional as F

def knn_predict(X_train, y_train, X_test, k=5, metric='l2'):
    """
    k-NN classifier using vectorised distance computation.
    X_train: (N, D), X_test: (M, D), y_train: (N,)
    Returns: predictions (M,)
    """
    # Compute pairwise distances between all test and train points
    if metric == 'l2':
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        train_sq = (X_train ** 2).sum(1, keepdim=True)  # (N, 1)
        test_sq  = (X_test  ** 2).sum(1, keepdim=True)  # (M, 1)
        cross    = X_test @ X_train.T                    # (M, N)
        dists    = test_sq + train_sq.T - 2 * cross      # (M, N)
    elif metric == 'l1':
        dists = torch.cdist(X_test, X_train, p=1)        # (M, N)

    # For each test point, find k nearest training points
    _, knn_idx = dists.topk(k, largest=False, dim=1)    # (M, k)
    knn_labels = y_train[knn_idx]                        # (M, k)

    # Majority vote
    preds = knn_labels.mode(dim=1).values               # (M,)
    return preds

# Demo
torch.manual_seed(0)
X_train = torch.randn(100, 50)
y_train = torch.randint(0, 5, (100,))
X_test  = torch.randn(20,  50)
y_test  = torch.randint(0, 5, (20,))

preds = knn_predict(X_train, y_train, X_test, k=5)
acc   = (preds == y_test).float().mean()
print(f'k-NN accuracy (random features, k=5): {acc:.2%}')
# On random features this is near chance (20% for 5 classes)
# On CNN features this would be much higher
```

---

## 7  K-Means Clustering

K-means is an unsupervised algorithm that partitions $n$ points into $k$ clusters, minimising the within-cluster sum of squared distances to the cluster centre (centroid):

$$\text{Objective:} \quad \min_{\{\mu_j\}, \{c^{(i)}\}} \sum_{i=1}^{n} \| \mathbf{x}^{(i)} - \mu_{c^{(i)}} \|^2$$

where $\mu_j$ is the centroid of cluster $j$ and $c^{(i)}$ is the cluster assignment of point $i$.

**Algorithm:**

1. **Initialise:** choose $k$ random centroids (or a random subset of data points).
2. **Assign:** assign each point to its nearest centroid: $c^{(i)} = \arg\min_j \|\mathbf{x}^{(i)} - \mu_j\|^2$.
3. **Update:** recompute each centroid as the mean of its assigned points: $\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{x}^{(i)}$.
4. **Repeat** steps 2–3 until assignments do not change (convergence).

K-means always converges (the objective is non-increasing) but may converge to a local minimum. Running with multiple random initialisations and keeping the best result (lowest objective) is standard practice.

```python
import torch

def kmeans(X, k, n_iters=100):
    """
    K-means clustering.
    X: (N, D) tensor of data points.
    Returns: centroids (k, D), assignments (N,)
    """
    N, D = X.shape
    # Initialise centroids from a random subset of data points
    idx       = torch.randperm(N)[:k]
    centroids = X[idx].clone()

    for _ in range(n_iters):
        # Step 1: Assign each point to nearest centroid
        dists   = torch.cdist(X, centroids)      # (N, k)
        assigns = dists.argmin(dim=1)            # (N,)

        # Step 2: Recompute centroids as cluster means
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assigns == j)
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]   # keep old if empty cluster

        # Convergence check
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids, assigns

# Demo: cluster 200 points into 4 groups
torch.manual_seed(0)
# Create 4 well-separated Gaussian clusters
X = torch.cat([
    torch.randn(50, 2) + torch.tensor([0.0,  0.0]),
    torch.randn(50, 2) + torch.tensor([5.0,  0.0]),
    torch.randn(50, 2) + torch.tensor([0.0,  5.0]),
    torch.randn(50, 2) + torch.tensor([5.0,  5.0]),
])
centroids, assigns = kmeans(X, k=4)
print(f'Centroids found:\n{centroids.round(decimals=1)}')
# Should be close to [0,0], [5,0], [0,5], [5,5]
```

---

## 8  Summary

| Concept | Model | Loss | Key insight |
|---|---|---|---|
| Linear regression | $h_\mathbf{w}(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$ | L2 / MSE | Gradient is error × feature |
| Logistic regression | $h_\mathbf{w}(\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$ | Binary cross-entropy | MLE derivation; same gradient form |
| Softmax regression | $\text{softmax}(\mathbf{W}^T\mathbf{x})$ | Multi-class cross-entropy | One template per class |
| Weight decay | $J_\text{reg} = J + \lambda \|\mathbf{w}\|^2$ | — | Shrinks weights to prevent overfitting |
| k-NN | Non-parametric | — | Fast train, slow predict; needs good features |
| k-means | Centroid assignment | Within-cluster SS | Unsupervised; may converge to local min |

The thread running through all topics is **loss functions and gradient descent**. In later lectures, the models become much richer (deep neural networks instead of linear models), but the optimisation machinery — define a differentiable loss, compute its gradient, step in the direction of steepest descent — remains exactly the same.

## References

- Andrew Ng, Stanford CS229 notes (linear and logistic regression derivations): cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf
- Neural network playground (visualise linear vs non-linear classifiers): playground.tensorflow.org
- Karpathy's ConvNetJS MNIST demo: cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
