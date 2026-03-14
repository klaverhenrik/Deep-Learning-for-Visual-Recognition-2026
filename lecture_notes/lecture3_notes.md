# Lecture 3 — Neural Networks and Backpropagation

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes build neural networks from the ground up: starting from logistic regression as a single neuron, adding hidden layers, deriving the four backpropagation equations from first principles, and finishing with computational graphs and PyTorch autograd. The LSTM skip-connection analogy (ResNets solve the same problem in space that LSTMs solve in time) is flagged where relevant.

---

## 1  From Logistic Regression to Neural Networks

Softmax regression is powerful but fundamentally limited: it can only learn **linear decision boundaries**. Many real classification problems — including almost all image recognition tasks — require non-linear boundaries. The fix is to compose multiple linear transformations with non-linear activation functions.

A single neuron takes a weighted sum of its inputs and passes the result through a non-linearity:

$$a = \sigma\!\left(\mathbf{w}^T\mathbf{x} + b\right)$$

A **neural network** stacks many such neurons in layers. The outputs of one layer become the inputs of the next, and the composition of many simple non-linear functions can represent arbitrarily complex mappings (given enough neurons and layers).

---

## 2  Architecture

### 2.1  Layers

A fully-connected (dense) neural network with $L$ layers computes:

$$\mathbf{a}^{[0]} = \mathbf{x} \quad \text{(input)}$$

$$\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]} \mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad \ell = 1, \ldots, L$$

$$\mathbf{a}^{[\ell]} = g^{[\ell]}\!\left(\mathbf{z}^{[\ell]}\right)$$

where $\mathbf{W}^{[\ell]}$ and $\mathbf{b}^{[\ell]}$ are the weight matrix and bias vector of layer $\ell$, and $g^{[\ell]}$ is the activation function. The final layer output $\mathbf{a}^{[L]}$ is the network's prediction.

### 2.2  Notation

| Symbol | Meaning |
|---|---|
| $L$ | Number of layers |
| $n^{[\ell]}$ | Number of neurons in layer $\ell$ |
| $\mathbf{W}^{[\ell]} \in \mathbb{R}^{n^{[\ell]} \times n^{[\ell-1]}}$ | Weight matrix of layer $\ell$ |
| $\mathbf{b}^{[\ell]} \in \mathbb{R}^{n^{[\ell]}}$ | Bias vector of layer $\ell$ |
| $\mathbf{z}^{[\ell]}$ | Pre-activation (linear combination) |
| $\mathbf{a}^{[\ell]}$ | Post-activation (output of layer $\ell$) |

### 2.3  Activation Functions

Without non-linearities, any stack of linear layers collapses to a single linear transformation: $\mathbf{W}^{[L]} \cdots \mathbf{W}^{[1]} \mathbf{x}$. Non-linear activations break this degeneracy.

Common choices:

$$\text{Sigmoid:} \quad \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

$$\text{Tanh:} \quad \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$$

$$\text{ReLU:} \quad \text{ReLU}(z) = \max(0, z)$$

$$\text{Leaky ReLU:} \quad f(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha \ll 1$$

ReLU is the default choice for hidden layers in modern networks. Its derivative is exactly 1 for positive inputs (no gradient saturation) and 0 for negative inputs (dead neurons if too many units are always negative).

---

## 3  Loss Functions

### 3.1  Regression: Mean Squared Error

$$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left\| \hat{y}^{(i)} - y^{(i)} \right\|^2$$

### 3.2  Binary Classification: Binary Cross-Entropy

$$J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 3.3  Multi-class Classification: Categorical Cross-Entropy

$$J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_k^{(i)} \log \hat{y}_k^{(i)}$$

where $\hat{y}_k^{(i)} = \text{softmax}(\mathbf{z}^{[L]})_k = \frac{\exp(z_k^{[L]})}{\sum_j \exp(z_j^{[L]})}$.

---

## 4  Backpropagation

Backpropagation is the algorithm for computing $\nabla_\theta J$ efficiently — the gradient of the loss with respect to every parameter in the network. It applies the chain rule systematically, working backwards from the output layer to the input layer.

### 4.1  The Four Equations

Define the **error signal** at layer $\ell$ as:

$$\boldsymbol{\delta}^{[\ell]} = \frac{\partial J}{\partial \mathbf{z}^{[\ell]}}$$

This measures how sensitive the loss is to the pre-activations at layer $\ell$.

**Equation 1 — Error at output layer:**

$$\boldsymbol{\delta}^{[L]} = \frac{\partial J}{\partial \mathbf{a}^{[L]}} \odot g'^{[L]}\!\left(\mathbf{z}^{[L]}\right)$$

For softmax + cross-entropy this simplifies to $\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$ (predicted minus true).

**Equation 2 — Backpropagate error through layers:**

$$\boldsymbol{\delta}^{[\ell]} = \left(\mathbf{W}^{[\ell+1]}\right)^T \boldsymbol{\delta}^{[\ell+1]} \odot g'^{[\ell]}\!\left(\mathbf{z}^{[\ell]}\right)$$

The error signal is propagated backwards through the transpose of the weight matrix, then element-wise multiplied by the derivative of the activation function at that layer.

**Equation 3 — Gradient with respect to weights:**

$$\frac{\partial J}{\partial \mathbf{W}^{[\ell]}} = \boldsymbol{\delta}^{[\ell]} \left(\mathbf{a}^{[\ell-1]}\right)^T$$

**Equation 4 — Gradient with respect to biases:**

$$\frac{\partial J}{\partial \mathbf{b}^{[\ell]}} = \boldsymbol{\delta}^{[\ell]}$$

### 4.2  Why the Transpose?

Equation 2 uses $(\mathbf{W}^{[\ell+1]})^T$. Intuitively: during the forward pass, weight matrix $\mathbf{W}^{[\ell+1]}$ maps activations *forward* from layer $\ell$ to layer $\ell+1$. During backpropagation, error signals travel *backwards*, and the transpose maps them in the reverse direction, distributing each neuron's error contribution to the neurons that fed it.

```python
import torch
import torch.nn as nn

# ── Neural network with manual backprop vs autograd ──────────────────
torch.manual_seed(0)

# ── From-scratch implementation ──────────────────────────────────────
def sigmoid(z):       return 1 / (1 + torch.exp(-z))
def sigmoid_prime(z): return sigmoid(z) * (1 - sigmoid(z))
def relu(z):          return torch.clamp(z, min=0)
def relu_prime(z):    return (z > 0).float()

class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, lr=0.01):
        self.W1 = torch.randn(n_hidden, n_in)  * 0.01
        self.b1 = torch.zeros(n_hidden, 1)
        self.W2 = torch.randn(n_out, n_hidden) * 0.01
        self.b2 = torch.zeros(n_out, 1)
        self.lr = lr

    def forward(self, x):
        # x: (n_in, 1)
        self.z1 = self.W1 @ x + self.b1       # (n_hidden, 1)
        self.a1 = relu(self.z1)               # (n_hidden, 1)
        self.z2 = self.W2 @ self.a1 + self.b2 # (n_out, 1)
        self.a2 = sigmoid(self.z2)            # (n_out, 1)
        return self.a2

    def backward(self, x, y):
        # Equation 1: output layer error
        delta2 = self.a2 - y                  # dJ/da2 * sigmoid'(z2) simplified

        # Equation 3 & 4: gradients for layer 2
        dW2 = delta2 @ self.a1.T
        db2 = delta2

        # Equation 2: propagate error to layer 1
        delta1 = (self.W2.T @ delta2) * relu_prime(self.z1)

        # Equation 3 & 4: gradients for layer 1
        dW1 = delta1 @ x.T
        db1 = delta1

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

# ── Verify against PyTorch autograd ──────────────────────────────────
net = TwoLayerNet(4, 8, 1, lr=0.01)
x   = torch.randn(4, 1)
y   = torch.tensor([[1.0]])

pred_manual = net.forward(x)
loss_manual = -y * torch.log(pred_manual + 1e-8) - (1-y)*torch.log(1-pred_manual+1e-8)
net.backward(x, y)
print(f'Manual forward loss: {loss_manual.item():.4f}')

# Same network in PyTorch autograd
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
x_t   = x.T                             # (1, 4) for nn.Linear
y_t   = y.T
loss  = nn.BCELoss()(model(x_t), y_t)
loss.backward()
print(f'Autograd loss: {loss.item():.4f}')
```

---

## 5  Computational Graphs and Autograd

A **computational graph** is a directed acyclic graph where:
- **Nodes** represent operations or variables
- **Edges** represent the flow of data (tensors)

Every tensor operation in PyTorch is recorded in a computational graph. Calling `.backward()` on a scalar loss traverses this graph in reverse, applying the chain rule at each node to accumulate gradients. This is exactly the backpropagation algorithm, automated.

```python
import torch

# ── Tracing the computational graph ──────────────────────────────────
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, 1.0], requires_grad=True)

# Forward pass — PyTorch builds the graph silently
z    = (w * x).sum()         # z = 0.5*2 + 1.0*3 = 4.0
loss = z ** 2                # loss = 16.0

# Backward pass — chain rule through the graph
loss.backward()

print(f'z    = {z.item()}')          # 4.0
print(f'loss = {loss.item()}')       # 16.0
print(f'dL/dw = {w.grad}')          # dL/dw = 2z * x = 8 * [2,3] = [16, 24]
print(f'dL/dx = {x.grad}')          # dL/dx = 2z * w = 8 * [0.5,1] = [4, 8]

# ── gradient_fn shows the graph structure ────────────────────────────
print(f'loss.grad_fn = {loss.grad_fn}')   # PowBackward0
print(f'z.grad_fn    = {z.grad_fn}')      # SumBackward0

# ── torch.no_grad() disables graph construction (inference) ──────────
with torch.no_grad():
    z_no_grad = (w * x).sum()
    print(f'z_no_grad.grad_fn = {z_no_grad.grad_fn}')  # None
```

---

## 6  Practical Training

### 6.1  Weight Initialisation

Never initialise all weights to zero: every neuron in a layer would compute the same gradient and remain identical forever (symmetry breaking failure). Initialise with small random values. For networks with ReLU activations, **Kaiming (He) initialisation** is the standard:

$$w \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{\text{in}}}\right)$$

### 6.2  Mini-batch SGD

Full-batch gradient descent computes the gradient over the entire dataset per update step — expensive. **Mini-batch SGD** uses a random subset (batch) of $B$ examples per step:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\mathbf{w} J^{(i)}$$

Typical batch sizes: 32–512. Smaller batches add noise (acts as regularisation); larger batches give more accurate gradient estimates but require more memory.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Complete training loop ────────────────────────────────────────────
torch.manual_seed(0)

# Synthetic dataset: XOR-like non-linear problem
n = 1000
X = torch.randn(n, 2)
y = ((X[:, 0] * X[:, 1]) > 0).float()

dataset    = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Sigmoid()
)

# Kaiming init for ReLU layers (PyTorch default for nn.Linear is Kaiming uniform)
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(20):
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        pred  = model(X_batch).squeeze(1)
        loss  = loss_fn(pred, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if epoch % 5 == 0:
        print(f'Epoch {epoch:3d}  loss: {epoch_loss/len(dataloader):.4f}')

with torch.no_grad():
    acc = ((model(X).squeeze(1) > 0.5) == y).float().mean()
    print(f'Final accuracy: {acc:.2%}')
```

---

## 7  Summary

| Concept | Key equation | Notes |
|---|---|---|
| Forward pass | $\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]}\mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}$, $\mathbf{a}^{[\ell]} = g(\mathbf{z}^{[\ell]})$ | Same for every layer |
| Output error | $\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$ | For softmax + cross-entropy |
| Backprop | $\boldsymbol{\delta}^{[\ell]} = (\mathbf{W}^{[\ell+1]})^T \boldsymbol{\delta}^{[\ell+1]} \odot g'(\mathbf{z}^{[\ell]})$ | Chain rule in matrix form |
| Weight gradient | $\partial J / \partial \mathbf{W}^{[\ell]} = \boldsymbol{\delta}^{[\ell]} (\mathbf{a}^{[\ell-1]})^T$ | Outer product |
| Bias gradient | $\partial J / \partial \mathbf{b}^{[\ell]} = \boldsymbol{\delta}^{[\ell]}$ | Same as error signal |

Backpropagation is not a special algorithm — it is the **chain rule of calculus applied to computational graphs**, automated. PyTorch's autograd builds the graph during the forward pass and traverses it in reverse during `.backward()`. Understanding the four equations above is what lets you debug gradient flow problems in deeper architectures.

## References

- Nielsen, M. (2015). Neural Networks and Deep Learning: neuralnetworksanddeeplearning.com
- Karpathy, A. Hacker's guide to Neural Networks: karpathy.github.io/neuralnets/
- Stanford CS229 notes: cs229.stanford.edu
