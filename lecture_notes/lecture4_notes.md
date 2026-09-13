# Lecture 4
# Convolutional Neural Networks

*Deep Learning for Visual Recognition · Aarhus University*

These notes accompany the lecture slides and provide narrative explanations and PyTorch code to help map concepts to practice.

---

## 1  Motivation: Why Not Fully Connected Networks?

In Lecture 3 we built fully connected (FC) neural networks where every neuron in one layer is connected to every neuron in the next. For a $512 \times 512$ RGB image that means the first hidden layer already has $512 \times 512 \times 3 = 786{,}432$ inputs per neuron. A single hidden layer with just 1,000 neurons would therefore require nearly 800 million weights — before training has even started. This is clearly impractical, but the problem goes deeper than memory.

The key insight is that image features are local. An edge, a corner, or a texture patch occupies only a small region of an image. A neuron connected to every pixel in the image has no natural way to specialise in such a local pattern: it has to learn to ignore all-but-75 of its 786,432 weights, which is a waste of capacity and data. Convolutional Neural Networks (CNNs) are designed specifically to exploit this locality.

> **Key idea.** Instead of learning one large weight matrix, a CNN learns a small filter (e.g. $5 \times 5 \times 3 = 75$ weights) and slides it over the entire image. The same filter detects the same feature — an edge, a curve — wherever it appears. This is the idea behind parameter sharing.

---

## 2  Convolution as Feature Extraction

### 2.1  What Is Convolution?

Formally, the 2-D discrete convolution of an input image $f$ with a filter $g$ at position $(x, y)$ is:

$$(f * g)(x, y) = \sum_m \sum_n f(x-m,\, y-n) \cdot g(m, n)$$

In practice, deep learning frameworks implement cross-correlation (the filter is not flipped), but the distinction does not matter for learning because the filter weights are learned anyway.

Each output value is a weighted sum of a small neighbourhood of pixels, where the weights are the filter values. Different filters detect different patterns:

- A [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator) (hand-crafted) detects horizontal or vertical edges.
- A [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_filter) (hand-crafted) smooths the image by averaging nearby pixels.
- In a CNN, the filters are not hand-crafted — they are learned from data by gradient descent.

### 2.2  Convolution in a Neural Network

Each output neuron in a CNN's feature map computes:

$$\text{output} = \text{activation}(\mathbf{w}^T \mathbf{x} + b)$$

where $\mathbf{w}$ is the filter (a small vector of learned weights), $\mathbf{x}$ is the flattened patch of input pixels under the filter, and $b$ is a bias term. The inner product $\mathbf{w}^T \mathbf{x}$ is simply a measure of how much the local image patch resembles the pattern encoded in the filter. When the patch matches the filter's pattern, the dot product is large and the neuron fires strongly.

> **PyTorch mapping.** In PyTorch, a convolutional layer is `nn.Conv2d(in_channels, out_channels, kernel_size)`. Each `out_channel` corresponds to one learned filter. The weight tensor has shape `(out_channels, in_channels, kH, kW)`.

```python
import torch
import torch.nn as nn

# A single convolutional layer:
#   - 3 input channels (RGB)
#   - 16 output channels (= 16 learned filters)
#   - 5x5 spatial kernel
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)

# Weight shape: (out_channels, in_channels, kH, kW)
print(conv.weight.shape)   # → torch.Size([16, 3, 5, 5])
print(conv.bias.shape)     # → torch.Size([16])  (one bias per filter)

# Forward pass on a batch of 8 images, each 3×64×64
x = torch.randn(8, 3, 64, 64)   # (batch, channels, H, W)
out = conv(x)
print(out.shape)   # → torch.Size([8, 16, 60, 60])  (no padding, stride=1)
```

*Code 1 – A minimal convolutional layer in PyTorch. Notice that the spatial output size shrinks from 64 to $60 = 64 - 5 + 1$ (no padding, stride 1).*

---

## 3  Three Key Properties of Convolution

### 3.1  Sparse Interactions

In a fully connected layer with $n^2$ input pixels and $(n-k)^2$ output neurons, the weight matrix has $n^2 \cdot (n-k)^2$ entries. For a $512 \times 512$ image, this is astronomical.

A convolutional layer with $m$ filters of size $k \times k$ has only $m \cdot k^2$ weights — independent of the image size. For $m = 64$, $k = 3$, that is just 576 weights, regardless of whether the image is $64 \times 64$ or $4096 \times 4096$.

Each output neuron is connected to only $k \times k \times C_\text{in}$ input values (where $C_\text{in}$ is the number of input channels), not to the entire image. This is what 'sparse interactions' means.

### 3.2  Parameter Sharing

The same filter weights are used at every spatial location of the image. So when the network learns a filter that detects, say, a 45-degree edge, that detector is automatically applied everywhere. There is no need to learn a separate edge detector for the top-left corner, the centre, and the bottom-right corner of the image.

During backpropagation, the gradient for a shared weight is the sum of gradients contributed by every spatial position where that filter was applied. This makes learning very data-efficient: a single training image contributes many gradient signals for each filter weight.

### 3.3  Translation Equivariance (and Approximate Invariance)

If an object shifts by $(\Delta x, \Delta y)$ pixels, the feature map produced by a convolutional layer shifts by the same amount. Formally, with $f$ denoting convolution and $T$ a translation operator:

$$f(T(\mathbf{x})) = T(f(\mathbf{x}))$$

This is equivariance, not invariance: the response shifts along with the object. Max pooling (Section 5) then adds a degree of invariance — small shifts in the feature map get absorbed by the pooling operation, so the pooled output stays approximately the same.

---

## 4  Feature Maps, Volumes, and the CNN Architecture

### 4.1  From Vectors to Volumes

Fully connected networks work on vectors. CNNs work on volumes — 3-D arrays with dimensions (Height × Width × Channels). An RGB image of size $512 \times 512$ is a volume of shape $512 \times 512 \times 3$.

After each convolutional layer, the output is also a volume: its spatial dimensions depend on the input size, stride, and padding, while its depth equals the number of filters. Each 'slice' of this output volume — one channel — is called a feature map or activation map. It shows where in the image a particular filter responds strongly.

```python
# Visualising the shapes through a small CNN
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32 filters, 3x3
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 filters, 3x3
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),        # halves H and W
)

x = torch.randn(1, 3, 64, 64)   # 1 image, RGB, 64x64
for layer in model:
    x = layer(x)
    print(f'{layer.__class__.__name__:20s}  →  {tuple(x.shape)}')

# Output:
# Conv2d                →  (1, 32, 64, 64)
# ReLU                  →  (1, 32, 64, 64)
# Conv2d                →  (1, 64, 64, 64)
# ReLU                  →  (1, 64, 64, 64)
# MaxPool2d             →  (1, 64, 32, 32)
```

*Code 2 – Tracing shapes through a small CNN. With `padding=1` and a $3 \times 3$ kernel, the spatial size is preserved (same convolution). `MaxPool2d` halves the spatial size.*

### 4.2  Receptive Field

The receptive field of a neuron is the region of the original input image that can influence its activation. In the first convolutional layer, a neuron with a $5 \times 5$ kernel has a receptive field of $5 \times 5$ pixels. After a max pooling layer that halves spatial resolution, a neuron in the next layer's $5 \times 5$ filter covers a $10 \times 10$ region of the original image.

As we go deeper into a CNN, receptive fields grow. Neurons in later layers respond to increasingly large regions of the input, allowing the network to detect large-scale, semantic features (object parts, whole objects) on top of the low-level features (edges, textures) detected in early layers.

> **Intuition across depth.** Early layers detect edges and blobs. Middle layers combine those into textures and object parts. Late layers encode high-level semantics like 'wheel', 'eye', or 'window'. This hierarchical feature learning is one of the core strengths of CNNs.

---

## 5  Controlling Spatial Dimensions: Stride and Padding

### 5.1  Output Size Formula

Given an input of spatial size $N$, a filter of size $W$, stride $S$, and zero-padding $P$ on each side, the output spatial size is:

$$\text{output\_size} = \left\lfloor \frac{N + 2P - W}{S} \right\rfloor + 1$$

Without padding ($P = 0$) and with stride $S = 1$, this simplifies to $N - W + 1$ (the output is smaller than the input). With 'same' padding ($P = (W-1)/2$) and $S = 1$, the output matches the input size exactly.

```python
import torch
import torch.nn as nn

# Manually verifying the output size formula
# Input: N=32, kernel W=5, stride S=1, padding P=0
# Expected output: (32 - 5) / 1 + 1 = 28

conv_valid = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)  # 'valid'
conv_same  = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)  # 'same'

x = torch.randn(1, 1, 32, 32)  # Input: 1 image, 1 channel, 32x32

print(conv_valid(x).shape)   # → torch.Size([1, 1, 28, 28])
print(conv_same(x).shape)    # → torch.Size([1, 1, 32, 32])

# With stride=2, 'valid' convolution halves spatial size
conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
print(conv_stride2(x).shape) # → torch.Size([1, 1, 15, 15])
# formula: (32 - 3) / 2 + 1 = 15.5 → floor = 15
```

*Code 3 – Verifying the output size formula. In PyTorch, `padding=0` corresponds to 'valid' and `padding=(kernel_size-1)//2` gives 'same' convolution for odd kernel sizes.*

### 5.2  Zero Padding

Without padding, every convolution shrinks the feature map. Stack enough layers and the spatial size goes to zero before the network is deep enough to be useful. Zero padding addresses this in two ways:

- It preserves the spatial size (with 'same' padding), allowing very deep architectures.
- It prevents 'border washing': without padding, border pixels contribute to fewer output positions than central pixels and are effectively under-represented in the feature maps.

In PyTorch, `padding=0` is 'valid' (no padding) and `padding=(kernel_size-1)//2` gives 'same' convolution for odd-sized kernels. Since PyTorch 1.9, `nn.Conv2d` also accepts the string literals `'valid'` and `'same'` directly for `padding` (for `stride=1`), which maps onto exactly the same terminology used by other frameworks:

```python
import torch
import torch.nn as nn

x = torch.randn(1, 1, 32, 32)

# The two literal padding modes, spelled out exactly as the terminology
# is normally introduced: 'valid' = no padding, 'same' = output size
# matches input size.
conv_valid = nn.Conv2d(1, 1, kernel_size=5, padding='valid')
conv_same  = nn.Conv2d(1, 1, kernel_size=5, padding='same')

print('valid:', conv_valid(x).shape)   # → torch.Size([1, 1, 28, 28])  (shrinks)
print('same: ', conv_same(x).shape)    # → torch.Size([1, 1, 32, 32])  (unchanged)
```

*Code 3b – The literal `padding='valid'` and `padding='same'` strings are only valid for `stride=1` — for a strided convolution you must fall back to an explicit integer (as in Code 3 above) and compute the padding yourself with the output-size formula.*

---

## 6  Activation Functions: Why We Need Non-Linearity

Convolution is a linear operation — it is equivalent to a matrix-vector multiplication. A stack of purely linear layers collapses to a single linear transformation, no matter how deep. To learn non-linear decision boundaries, we need non-linear activation functions between layers.

Modern CNNs use ReLU (Rectified Linear Unit) by default:

$$\text{ReLU}(z) = \max(0, z)$$

ReLU has two major advantages over the sigmoid and tanh activations used in earlier networks:

- It does not saturate for positive inputs — its gradient is a constant 1 — which avoids the vanishing gradient problem.
- It is computationally trivial (a single threshold operation), making networks faster to train. The original AlexNet paper noted roughly $6\times$ faster convergence with ReLU compared to tanh.

The downside is the dying ReLU problem: if a neuron's input is always negative, it always outputs zero and receives no gradient update. This is addressed by variants such as Leaky ReLU, ELU, or by careful weight initialisation (covered in Lecture 5).

```python
import torch
import torch.nn as nn

# ReLU applied element-wise after convolution
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
relu = nn.ReLU()

x = torch.randn(1, 3, 64, 64)
z = conv(x)      # pre-activation: can be any real value
a = relu(z)      # post-activation: non-negative

print(f'Pre-ReLU  min: {z.min():.2f}, max: {z.max():.2f}')
print(f'Post-ReLU min: {a.min():.2f}, max: {a.max():.2f}')  # min is 0

# Alternatively, use inplace=True to save memory
relu_inplace = nn.ReLU(inplace=True)
```

*Code 4 – Applying ReLU after a convolution. After ReLU, all negative pre-activations are clipped to zero.*

---

## 7  Pooling Layers

### 7.1  Max Pooling

A pooling layer reduces the spatial dimensions of a feature map by summarising a local region with a single value. Max pooling takes the maximum value in each local window. With a $2 \times 2$ window and stride 2 (the standard setting), the spatial dimensions are halved:

$$\text{MaxPool}(\mathbf{x})[i,j] = \max\bigl(x[2i,\,2j],\; x[2i+1,\,2j],\; x[2i,\,2j+1],\; x[2i+1,\,2j+1]\bigr)$$

Why maximum rather than average? The maximum preserves the presence of the strongest feature activation in the region, regardless of its exact position. This provides a degree of translation invariance: if the feature shifts by 1 pixel within the $2 \times 2$ window, the max pooling output is unchanged.

```python
import torch
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Demonstrate translation invariance within the pooling window
x1 = torch.tensor([[[[3., 0.],
                      [0., 0.]]]])   # strong activation top-left
x2 = torch.tensor([[[[0., 0.],
                      [0., 3.]]]])   # strong activation bottom-right

print(pool(x1))  # → tensor([[[[3.]]]])
print(pool(x2))  # → tensor([[[[3.]]]])  same output!

# In practice, pooling is applied across all channels independently
x = torch.randn(1, 64, 32, 32)
out = pool(x)
print(out.shape)   # → torch.Size([1, 64, 16, 16])  — halved spatially
```

*Code 5 – Max pooling. Both `x1` and `x2` produce the same output (3.0) despite the active pixel being in different corners, illustrating local translation invariance.*

### 7.2  Is Max Pooling the Right Choice?

Geoffrey Hinton has argued that max pooling discards exact positional information, which could be important for tasks like face recognition (where the relative positions of eyes, nose, and mouth matter). One practical alternative is to replace max pooling with a strided convolution:

- `nn.MaxPool2d(2, 2)` — discards the location of the winning activation.
- `nn.Conv2d(..., stride=2)` — downsamples by learning which features to keep, retaining more information.

Many modern architectures (e.g., ResNets, most Vision Transformers) use strided convolutions instead of max pooling for this reason.

```python
import torch
import torch.nn as nn

# Max pooling vs. strided convolution for downsampling

# Option 1: Max pooling (no learnable parameters)
downsample_pool = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
)

# Option 2: Strided convolution (learns how to downsample)
downsample_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

x = torch.randn(1, 64, 32, 32)
print(downsample_pool(x).shape)   # → (1, 64, 16, 16)
print(downsample_conv(x).shape)   # → (1, 64, 16, 16)
# Same output shape, but strided conv has learnable parameters
```

*Code 6 – Two approaches to spatial downsampling. Strided convolution has learnable parameters; max pooling does not.*

### 7.3  Global Average Pooling

After the convolutional layers, we need a fixed-length vector to feed into the fully connected classifier. The naive approach is to flatten the feature map — but then the vector length depends on the input image size, causing a mismatch if the image size changes.

Global Average Pooling (GAP) solves this by computing the mean of each entire feature map, producing exactly one number per channel. The result is always a vector of length $C$ (the number of channels), regardless of the spatial size of the feature map:

$$\text{GAP}(\text{feature\_map})[c] = \text{mean over all } H \times W \text{ positions of channel } c$$

```python
import torch
import torch.nn as nn

gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))   # Global Average Pooling

# Works for any spatial size
for h, w in [(7, 7), (3, 3), (14, 14)]:
    x = torch.randn(1, 1024, h, w)
    out = gap(x)
    print(f'Input: (1, 1024, {h}, {w})  →  GAP output: {tuple(out.shape)}')

# All produce: (1, 1024, 1, 1) — then flatten to (1, 1024)

# Full pattern: conv base → GAP → flatten → classifier
model = nn.Sequential(
    nn.Conv2d(3, 1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),   # GAP
    nn.Flatten(),                   # (batch, 1024)
    nn.Linear(1024, 10),            # classifier for 10 classes
)
```

*Code 7 – Global Average Pooling with `nn.AdaptiveAvgPool2d`. The output is always `(batch, C, 1, 1)` regardless of input spatial size. After flattening, this becomes a fixed-length vector of size $C$.*

---

## 8  Fully Connected Layers and 1×1 Convolutions

### 8.1  The Fully Connected Layer

After the convolutional base, most CNN classifiers end with one or more fully connected (FC) layers (also called dense or linear layers in PyTorch's terminology). These are standard matrix-vector products:

$$\mathbf{x}_\text{out} = \text{activation}(\mathbf{W} \cdot \mathbf{x}_\text{in} + \mathbf{b})$$

A fully connected layer requires its input to be a vector. The 3-D output of the last convolutional/pooling layer must be flattened first (e.g., a $6 \times 6 \times 256$ feature map becomes a vector of length 9,216). In PyTorch this is done with `nn.Flatten()`.

```python
import torch
import torch.nn as nn

# The flattening step before a fully connected layer
x_volume = torch.randn(8, 256, 6, 6)   # batch of 8 feature maps (256 channels, 6x6 spatial)

flatten = nn.Flatten()                 # default: flattens dims 1 onwards
x_flat  = flatten(x_volume)
print(x_flat.shape)   # → torch.Size([8, 9216])  (8 × 256×6×6)

fc = nn.Linear(9216, 4096)
out = fc(x_flat)
print(out.shape)      # → torch.Size([8, 4096])
```

*Code 8 – Flattening a 3-D feature map before a fully connected layer. The flatten operation turns `(batch, C, H, W)` into `(batch, C×H×W)`.*

### 8.2  1×1 Convolutions: A Powerful Alternative

A $1 \times 1$ convolutional filter applies a learned linear combination across channels at every spatial position independently. This is equivalent to running a small fully connected network at each pixel:

- Input: feature map of shape $(H, W, C_\text{in})$
- $1 \times 1$ convolution with $C_\text{out}$ filters: output has shape $(H, W, C_\text{out})$
- For each of the $H \times W$ spatial positions, the filter computes a $C_\text{in} \to C_\text{out}$ linear projection

One particularly powerful consequence: if we replace the final FC layer with $1 \times 1$ convolutions followed by GAP, the resulting network accepts inputs of any spatial size — the $1 \times 1$ conv operates independently at each position, so the spatial size never causes a shape mismatch. This is the basis of Fully Convolutional Networks (FCNs) used in segmentation.

```python
import torch
import torch.nn as nn

# 1x1 convolution for channel-wise dimensionality reduction
# Reduces from 1024 to 256 channels at every spatial position
pointwise = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)

x = torch.randn(1, 1024, 14, 14)  # 1 image, 1024 channels, 14x14 spatial
out = pointwise(x)
print(out.shape)   # → torch.Size([1, 256, 14, 14])

# Compare parameter counts:
# FC layer mapping 1024-dim to 256-dim:
fc = nn.Linear(1024, 256)
print(sum(p.numel() for p in fc.parameters()))         # 1024*256 + 256 = 262400

# 1x1 conv mapping 1024 channels to 256 channels:
print(sum(p.numel() for p in pointwise.parameters()))  # same! 1024*256 + 256 = 262400

# They have the same number of parameters — but 1x1 conv can handle any H×W
```

*Code 9 – $1 \times 1$ convolution as a channel-wise linear projection. It has the same parameter count as an equivalent FC layer but operates spatially and handles arbitrary input sizes.*

---

## 9  Classic CNN Architectures: LeNet-5 and AlexNet

### 9.1  LeNet-5 (1998)

LeNet-5, introduced by LeCun et al. in 1998, was the first successful CNN and was trained to recognise handwritten digits (MNIST). Its architecture is [CONV → POOL → CONV → POOL → FC → FC → FC]. Although small by modern standards, it established all the core patterns used today.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """LeNet-5 adapted for 32x32 grayscale input, 10 classes."""
    def __init__(self):
        super().__init__()
        # Convolutional base
        self.conv1 = nn.Conv2d(1, 6,  kernel_size=5)  # 32→28, 6 filters
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14→10, 16 filters
        # Classifier
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):             # x: (batch, 1, 32, 32)
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)  # → (batch, 6, 14, 14)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)  # → (batch, 16, 5, 5)
        x = x.flatten(start_dim=1)                  # → (batch, 400)
        x = F.relu(self.fc1(x))                     # → (batch, 120)
        x = F.relu(self.fc2(x))                     # → (batch, 84)
        x = self.fc3(x)                             # → (batch, 10)
        return x

model = LeNet5()
x = torch.randn(4, 1, 32, 32) # 4 images, 1 channel, 32x32
print(model(x).shape)  # → torch.Size([4, 10])

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total:,}')   # ≈ 61,706
```

*Code 10 – LeNet-5 implemented in PyTorch. The entire network has about 60,000 parameters — tiny by modern standards, but it established the CONV–POOL–FC blueprint.*

### 9.1.1  Inspecting a Model Layer by Layer

Keras provides `model.summary()` out of the box: a table listing every layer's type, output shape, and parameter count. PyTorch has no built-in equivalent, but it is easy to build one using forward hooks — functions that PyTorch calls automatically every time a given submodule finishes its forward pass. Registering a hook on every leaf submodule (a module with no children of its own, e.g. `Conv2d`, `Linear`) lets us record each one's output shape and parameter count with a single dummy forward pass:

```python
import torch

def summarize(model, input_size, batch_size=1):
    """A minimal PyTorch equivalent of Keras' model.summary()."""
    rows = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, module.__class__.__name__, tuple(output.shape), n_params))
        return hook

    # Only leaf modules (no children) — otherwise a container like
    # `self.features` would also fire its own hook and double-count.
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(batch_size, *input_size))
    model.train(was_training)

    for h in handles:
        h.remove()

    name_w  = max(len(r[0]) for r in rows) + 2
    type_w  = max(len(r[1]) for r in rows) + 2
    shape_w = max(len(str(r[2])) for r in rows) + 2
    header = f"{'Layer':{name_w}}{'Type':{type_w}}{'Output Shape':{shape_w}}{'Param #'}"
    print(header)
    print('-' * len(header))
    total = 0
    for name, ltype, shape, n in rows:
        total += n
        print(f"{name:{name_w}}{ltype:{type_w}}{str(shape):{shape_w}}{n:,}")
    print('-' * len(header))
    print(f"Total parameters: {total:,}")
    return rows

model = LeNet5()
summarize(model, input_size=(1, 32, 32))
```

```text
Layer  Type    Output Shape     Param #
---------------------------------------
conv1  Conv2d  (1, 6, 28, 28)   156
conv2  Conv2d  (1, 16, 10, 10)  2,416
fc1    Linear  (1, 120)         48,120
fc2    Linear  (1, 84)          10,164
fc3    Linear  (1, 10)          850
---------------------------------------
Total parameters: 61,706
```

*Code 10b – A hook-based `summarize()` utility applied to `LeNet5`. Notice what is **missing** compared to a Keras summary of an equivalent model: there are no rows for the pooling layers or the ReLU activations. This is because `LeNet5.forward()` calls `F.avg_pool2d(...)` and `F.relu(...)` as plain *functions*, not as submodules — and a forward hook can only fire on registered `nn.Module` children (here: `conv1`, `conv2`, `fc1`, `fc2`, `fc3`). If you want pooling or activation layers to appear as their own rows, define them as modules in `__init__` (e.g. `self.pool = nn.AvgPool2d(2)`) and call `self.pool(x)` in `forward()` instead of the functional form. This is a good illustration of a subtlety that trips people up when moving between frameworks: a Keras layer added with `model.add(...)` is always a module and always shows up in `model.summary()`, whereas the same operation in PyTorch may be written as a stateless function call that is invisible to any tool that only inspects registered submodules.*

### 9.2  AlexNet (2012)

AlexNet, winner of the ImageNet challenge in 2012, was essentially a deeper and wider LeNet trained on GPUs. Its key innovations were:

- ReLU activations (replacing sigmoid/tanh) — $6\times$ faster convergence.
- Training on two GPUs in parallel (a necessity at the time, now irrelevant).
- Data augmentation and dropout to combat overfitting.
- Local response normalisation (now superseded by batch normalisation).

AlexNet's architecture — five convolutional layers followed by three fully connected layers — was the template for most networks designed in the following years.

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """Simplified AlexNet for 224x224 RGB input, num_classes outputs."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 224→55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 55→27
            # Block 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),           # 27→27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 27→13
            # Block 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),          # 13→13
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),          # 13→13
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # 13→13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 13→6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)         # → (batch, 256, 6, 6)
        x = x.flatten(start_dim=1)   # → (batch, 9216)
        return self.classifier(x)    # → (batch, num_classes)

model = AlexNet(num_classes=10)
summarize(model, input_size=(3, 224, 224))
```

*Code 11 – AlexNet in PyTorch. The network has ~57 million parameters, the vast majority in the three large fully connected layers ($4096 \to 4096 \to 1000$). This is one motivation for replacing FC layers with GAP + smaller classifiers in later architectures.*

---

## 10  Putting It All Together: A Modern Minimal CNN

The code below builds a small but complete CNN for image classification, incorporating all the concepts from this lecture. Note how it follows the standard pattern: convolutional base (feature extraction) followed by a classifier head.

```python
import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """
    A compact CNN for image classification.
    Demonstrates all key components from Lecture 4.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # ── Convolutional base ────────────────────────────────────────
        # Pattern: Conv → ReLU → Conv → ReLU → MaxPool
        # Doubling channels after each pooling is a common design choice.
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels, spatial: 64 → 64 → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), # same conv
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halve spatial dims

            # Block 2: 32 → 64 channels, spatial: 32 → 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 → 128 channels, spatial: 16 → 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Classifier head ───────────────────────────────────────────
        # Global Average Pooling removes spatial dims entirely,
        # making the classifier work for any input image size.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # → (batch, 128, 1, 1)
            nn.Flatten(),                  # → (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # regularisation
            nn.Linear(64, num_classes),   # logits for each class
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Training loop skeleton ────────────────────────────────────────────
model     = SmallCNN(in_channels=3, num_classes=10)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

# Dummy batch: 16 RGB images of size 64x64
images = torch.randn(16, 3, 64, 64)  # (batch, channels, H, W)
labels = torch.randint(0, 10, (16,)) # random integer labels in [0, 9]

# One training step
optimiser.zero_grad()           # 1. Clear gradients
logits = model(images)          # 2. Forward pass
loss   = loss_fn(logits, labels)# 3. Compute loss
loss.backward()                 # 4. Backward pass (compute gradients)
optimiser.step()                # 5. Update weights

print(f'Loss: {loss.item():.4f}')
print(f'Output shape: {logits.shape}')   # → (16, 10)

# Count parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {n_params:,}')   # ≈ 102,602
```

*Code 12 – A complete small CNN including the training step. The five-line training loop (zero\_grad → forward → loss → backward → step) is the fundamental PyTorch training pattern you will use in every experiment.*

---

## 11  Transfer Learning

Training a CNN from scratch requires large datasets (ImageNet has 1.2 million images) and many GPU-hours. For most practical problems, training from scratch is unnecessary. Transfer learning reuses a network pre-trained on a large source dataset and adapts it to the target task.

There are two common scenarios:

- **Feature extraction**: Freeze the convolutional base and train only a new classifier head. Works well when the target dataset is small and similar to the source dataset.
- **Fine-tuning**: Load pre-trained weights, then train the entire network (or just the later layers) on the target dataset with a small learning rate. Works best when the target dataset is larger or significantly different from the source.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── Scenario 1: Feature extraction ───────────────────────────────────
# Load a ResNet-18 pre-trained on ImageNet
backbone = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all parameters so they are not updated during training
for param in backbone.parameters():
    param.requires_grad = False

# Replace the final classifier with one suited to our task (e.g., 5 classes)
backbone.fc = nn.Linear(backbone.fc.in_features, 5)  # only this is trained

# Verify: only the new head has trainable parameters
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f'Trainable (feature extraction): {trainable:,}')   # = 2,565  (just the new FC layer)


# ── Scenario 2: Fine-tuning ───────────────────────────────────────────
backbone2 = models.resnet18(weights='IMAGENET1K_V1')
backbone2.fc = nn.Linear(backbone2.fc.in_features, 5)

# Unfreeze everything — use a small learning rate to avoid destroying
# the pre-trained features
optimiser = torch.optim.Adam(backbone2.parameters(), lr=1e-4)

print(f'Trainable (fine-tuning): {sum(p.numel() for p in backbone2.parameters() if p.requires_grad):,}')   # = 11,179,077 (all parameters trainable)
```

*Code 13 – Transfer learning with a pre-trained ResNet-18. In Scenario 1 (feature extraction), freezing the backbone means only the new head's 2,565 parameters are trained. In Scenario 2 (fine-tuning), all ~11 million parameters are updated.*

---

## 12  Summary

The table below maps each concept from the lecture to its PyTorch equivalent:

| Concept | What it does | PyTorch |
|---|---|---|
| Convolutional layer | Applies learned filters to detect local features | `nn.Conv2d(C_in, C_out, kernel_size)` |
| ReLU activation | Adds non-linearity, avoids saturation | `nn.ReLU()` |
| Max pooling | Downsamples, adds translation invariance | `nn.MaxPool2d(2, 2)` |
| Stride | Controls step size and output resolution | `stride=` argument in `Conv2d` |
| Same padding | Preserves spatial size | `padding=(k-1)//2` in `Conv2d` |
| Flatten | Converts volume to vector for FC layers | `nn.Flatten()` |
| Global Avg Pool | Fixed-length representation for any input size | `nn.AdaptiveAvgPool2d((1,1))` |
| $1 \times 1$ convolution | Channel mixing / dimensionality reduction | `nn.Conv2d(C_in, C_out, 1)` |
| Fully connected | Linear classifier head | `nn.Linear(n_in, n_out)` |
| Transfer learning | Reuse pre-trained weights | `torchvision.models` + freeze params |

The core message of this lecture is simple: CNNs are efficient because they exploit the local structure and spatial regularity of images. Parameter sharing means that a network needs far fewer weights than a fully connected alternative. Translation equivariance means that the same feature detector works everywhere in the image. These two properties, combined with learned non-linear activations and pooling, are what made CNNs the dominant approach to visual recognition from 2012 onwards — and they remain central building blocks in modern architectures.

---

## 13  Exercises

### Exercise 1 — Output Size Formula

**(a)** An input of spatial size $N = 32$, convolved with a kernel of size $W = 5$, stride $S = 1$, and no padding ($P = 0$). Use the formula $\lfloor (N + 2P - W) / S \rfloor + 1$ to compute the output size.

**(b)** Repeat for $S = 2$, $P = 2$. What happens to the spatial size compared to the input?

**(c)** You want the output to have the same spatial size as the input when using a $3 \times 3$ kernel and stride 1. How much padding $P$ is needed? Generalise: for a kernel of size $W$ and stride 1, what padding preserves spatial size?

**(d)** A network applies three consecutive convolutions, each with kernel size 3, stride 1, and no padding, to a $32 \times 32$ input. What is the spatial size after all three convolutions? What is the receptive field of a single output neuron after all three layers?

---

### Exercise 2 — Parameter Counting in Convolutional Layers

**(a)** A convolutional layer has 3 input channels, 16 output channels, and a $5 \times 5$ kernel. How many learnable parameters does it have (weights + biases)?

**(b)** A fully connected layer takes the same $3 \times 32 \times 32$ volume as input and produces 16 outputs. How many parameters does it have? What is the ratio of FC to conv parameters?

**(c)** A conv layer has 64 input channels, 128 output channels, and a $3 \times 3$ kernel. How many parameters does it have? How does this change if the input image is $64 \times 64$ versus $256 \times 256$?

**(d)** Explain in one sentence why the answer to (c) does not depend on the spatial size of the input.

---

### Exercise 3 — Reading PyTorch Code: Tracing Shapes

```python
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 10),
)
```

**(a)** What is the shape of the output after each layer for an input of shape `(1, 1, 28, 28)`? Fill in the table:

| Layer | Output shape |
|-------|-------------|
| `Conv2d(1, 32, 3, padding=1)` | |
| `ReLU` | |
| `MaxPool2d(2, 2)` | |
| `Conv2d(32, 64, 3, padding=0)` | |
| `ReLU` | |
| `MaxPool2d(2, 2)` | |
| `Flatten` | |
| `Linear(64×6×6, 10)` | |

**(b)** The `nn.Linear` layer is hard-coded with input size `64 * 6 * 6`. What would break if you fed the model a `(1, 1, 32, 32)` image instead? How would you fix this?

**(c)** How many trainable parameters does the entire model have?

---

### Exercise 4 — Max Pooling and Translation Invariance

**(a)** Apply $2 \times 2$ max pooling with stride 2 to the following $4 \times 4$ feature map:

$$\begin{bmatrix} 1 & 3 & 2 & 4 \\ 5 & 6 & 1 & 2 \\ 7 & 2 & 8 & 3 \\ 1 & 4 & 6 & 5 \end{bmatrix}$$

What is the resulting $2 \times 2$ output?

**(b)** Now shift the strongest activation (the 8) one position to the right. Re-apply max pooling. Does the output change?

**(c)** Explain in two sentences what these results tell you about translation invariance versus translation equivariance, and which property max pooling adds.

**(d)** Geoffrey Hinton argued that max pooling discards positional information that might be important. Name one modern alternative to max pooling for downsampling, and explain how it differs.

---

### Exercise 5 — Receptive Field

**(a)** A single convolutional layer uses a $5 \times 5$ kernel. What is the receptive field of one output neuron?

**(b)** A second convolutional layer with a $5 \times 5$ kernel is added on top. What is the receptive field of a neuron in the second layer's output, measured in terms of the original input?

**(c)** Instead of one $5 \times 5$ conv, you use two stacked $3 \times 3$ convs (no pooling between them). What is the receptive field of a neuron after both layers? What is the parameter count of two $3 \times 3$ conv layers vs one $5 \times 5$ conv layer, for the same number of channels $C$?

**(d)** Explain why neurons in later layers of a deep CNN respond to semantically meaningful patterns (object parts, whole objects) rather than simple edges, even though each layer only uses small local filters.

---

### Exercise 6 — Theoretical: Key Properties of Convolution

**(a)** Define *sparse interactions* in the context of a convolutional layer. How does this differ from a fully connected layer?

**(b)** Define *parameter sharing*. What assumption about images does it encode, and why is this assumption generally valid for natural images?

**(c)** Distinguish between *translation equivariance* and *translation invariance*. Which property does a convolutional layer have on its own, and what is needed to achieve the other?

**(d)** A CNN is trained to classify whether an image contains a cat. The cat appears in the top-left corner of all training images. At test time the cat appears in the bottom-right corner. Would a CNN generalise better to this shift than a fully connected network of the same capacity? Explain why.

---

### Exercise 7 — Transfer Learning

```python
backbone = models.resnet18(weights='IMAGENET1K_V1')

for param in backbone.parameters():
    param.requires_grad = False

backbone.fc = nn.Linear(backbone.fc.in_features, 5)
optimiser = torch.optim.Adam(backbone.parameters(), lr=1e-3)
```

**(a)** How many classes does this model now predict? What happened to the original ImageNet classification head?

**(b)** After the loop that sets `requires_grad = False`, a new `backbone.fc` is assigned. Does the new FC layer have `requires_grad = True` or `False`? Why?

**(c)** A student argues that setting `requires_grad = False` on all parameters and then replacing `backbone.fc` means the optimiser will update nothing, since it was constructed before `backbone.fc` was replaced. Is this correct? Explain.

**(d)** You have a small medical imaging dataset (500 labelled images) that is quite different from ImageNet. Would you prefer feature extraction (frozen backbone) or full fine-tuning? Justify your answer.

## References

- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. NeurIPS.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 9 and 6.
- PyTorch documentation: https://pytorch.org/docs/stable/nn.html
- Stanford CS231n notes (Karpathy): https://cs231n.github.io/convolutional-networks/