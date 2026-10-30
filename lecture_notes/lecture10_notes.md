# Lecture 10
# Visualising and Understanding ConvNets

*Deep Learning for Visual Recognition · Aarhus University*

These notes cover every visualisation technique in the lecture — from simple filter and activation inspection through saliency maps and Grad-CAM to the full Neural Style Transfer pipeline. Every technique is grounded in the same fundamental operation: differentiating a loss with respect to pixels rather than weights, treating the image as the variable to optimise while the network stays frozen.

---

## 1  Why Visualise ConvNets?

Neural networks are often described as black boxes — you feed in an image, a class label comes out, and the internal computations are opaque. This is not only intellectually unsatisfying; it is practically dangerous. A classifier that achieves 95% accuracy on the test set might be doing so for entirely the wrong reasons — learning to recognise training artefacts, background correlations, or dataset biases rather than the true discriminative features of each class. The canonical example is a classifier that achieves high accuracy on a 'husky vs wolf' task not by distinguishing the animals but by detecting whether snow appears in the background, because most training wolves were photographed in snowy environments.

Visualisation techniques address this by answering questions such as: what does each neuron respond to? How does the network's representation of an image change across layers? Which regions of an input image are most responsible for a particular prediction? The answers guide architecture improvements, reveal training failures, and — as a beautiful bonus — enable entirely new creative applications like Neural Style Transfer.

The key challenge: neural networks are not invertible. Many different inputs can produce the same output, so there is no direct way to read off 'what the network is thinking' from its output. Every visualisation technique is therefore an approximation — a carefully designed question about the network's internals that can be answered indirectly.

> **The unifying theme of this lecture.** Almost every technique described here reduces to the same operation: run a forward pass through a frozen network, define a scalar loss based on some internal activation or output, and then differentiate that loss with respect to the input pixels using backpropagation. The result is a gradient map over pixels that answers a specific interpretability question. Neural Style Transfer, Grad-CAM, filter visualisation, and adversarial attacks are all instances of this pattern.

---

## 2  Visualising Activations and Filters

### 2.1  Layer Activations

The most direct visualisation is simply to plot the feature maps (activation maps) produced by each layer for a given input image. For a $224 \times 224$ RGB image passed through VGG16, the first convolutional layer (conv1\_1) produces 64 feature maps each of size $224 \times 224$ — one per filter. Plotting these 64 maps as greyscale images shows exactly what each filter has extracted from the input.

The pattern across layers is consistent and interpretable:

- **Early layers (conv1, conv2)**: Feature maps are dense and activate across most of the image. Filters respond to oriented edges, colour gradients, and simple blobs. Almost every region triggers some response because these primitive features appear everywhere in natural images.
- **Middle layers (conv3, conv4)**: Feature maps become progressively less interpretable visually. The network is building abstract representations — combinations of early features — that do not correspond to any easily named visual concept. The spatial extent of activations decreases as the receptive field grows.
- **Late layers (conv5)**: Many feature maps are entirely black (all zeros). This is not a problem — it means those filters are looking for complex patterns not present in the current image. A filter tuned to detect the texture of fur will not fire on an image of a bicycle.

A diagnostic use during training: for ReLU networks, healthy activations start broad and gradually become more sparse and localised as training progresses. Feature maps that are all-zero for many different inputs indicate dead ReLUs — a symptom of too-high learning rate (see Lecture 5).

### 2.2  Filter / Weight Visualisation

For the first convolutional layer only — where filters operate directly on raw RGB pixels — each filter can be visualised as a colour image (since it has exactly 3 channels, one per RGB colour). Well-trained networks show clean, smooth, structured filters: oriented Gabor-like edge detectors, colour blob detectors, frequency patterns. Noisy, grainy filters indicate poor convergence, too-low regularisation, or insufficient training time.

For deeper layers, filters have hundreds or thousands of channels and cannot be directly visualised as images. The reconstruction-based approach (Section 5) handles this.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, size=224, normalise=True):
    """Load an image as a (1, 3, size, size) tensor, preprocessed for ImageNet models."""
    tfms = [T.Resize(size), T.CenterCrop(size), T.ToTensor()]
    if normalise:
        tfms.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(tfms)(Image.open(path).convert('RGB')).unsqueeze(0)

def denormalise(x):
    """Undo the ImageNet normalisation so a tensor can be shown with plt.imshow."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1)

# ── Visualising layer activations with forward hooks ──────────────────
weights = models.VGG16_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)
model.eval()

# Make every ReLU a normal (not in-place) layer, so stored activations are never modified later
for m in model.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False

# Storage for captured activations
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Hook the ReLU layers: a ReLU output is exactly 0 wherever a neuron is inactive,
# whereas a conv output (before the ReLU) is almost never exactly 0.
model.features[1].register_forward_hook(make_hook('relu1_1'))    # after conv1_1: 64 filters
model.features[29].register_forward_hook(make_hook('relu5_3'))   # after conv5_3: 512 filters

# Run a forward pass on a real image
img = load_image('data/Cat03.jpg')
with torch.no_grad():
    logits = model(img)
print('Prediction:', weights.meta['categories'][logits.argmax().item()])

# Inspect the captured activations
for name, act in activations.items():
    frac_zero       = (act == 0).float().mean().item()                        # zero activations
    frac_black_maps = (act.flatten(2).amax(dim=2) == 0).float().mean().item() # completely black maps
    print(f'{name}: shape={tuple(act.shape)}  zeros={frac_zero:.1%}  all-zero feature maps={frac_black_maps:.1%}')

# ── Visualising first-layer filters directly ──────────────────────────
# First conv layer weights have shape (out_ch, in_ch, H, W) = (64, 3, 3, 3)
conv1_weights = model.features[0].weight.data.cpu()

def normalise_filter(w):
    w = w - w.min()
    return w / (w.max() + 1e-8)

fmap1 = activations['relu1_1'][0]                       # (64, 224, 224)
fmap5 = activations['relu5_3'][0]                       # (512, 14, 14)
top5  = fmap5.flatten(1).amax(dim=1).argsort(descending=True)[:8]   # 8 most active maps

plt.figure(figsize=(3, 3)); plt.imshow(denormalise(img)[0].permute(1, 2, 0)); plt.axis('off'); plt.title('Input')

fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
for i in range(8):
    axes[0, i].imshow(normalise_filter(conv1_weights[i]).permute(1, 2, 0), interpolation='nearest')
    axes[1, i].imshow(fmap1[i], cmap='gray')
    axes[2, i].imshow(fmap5[top5[i]], cmap='gray')
for ax in axes.ravel():
    ax.axis('off')
axes[0, 0].set_title('conv1_1 filters (3x3 RGB)', loc='left')
axes[1, 0].set_title('relu1_1 feature maps (early: dense, edge/colour-like)', loc='left')
axes[2, 0].set_title('relu5_3: 8 most active of 512 maps (late: sparse, localised)', loc='left')
plt.show()
```

*Code 1 – Capturing and inspecting layer activations using forward hooks. The hooks sit on the ReLU outputs, so the fraction of exact zeros (and of completely black feature maps) can be measured directly; for a given image, many zeros in a late layer are normal, but all-zero maps across many images flag dead ReLU problems. First-layer filters can be directly rendered as colour images ($3 \times 3$ RGB); deeper filters require the reconstruction-based approach in Section 5.*

---

## 3  Inspecting FC Layers: k-NN and t-SNE

### 3.1  Nearest Neighbours in Feature Space

Fully connected layers produce high-dimensional embeddings (e.g. 4096-d in AlexNet/VGGNet) that summarise an image's semantic content. To inspect what the network has learned, extract these embeddings for a large image collection and search for nearest neighbours using k-NN. If the network has learned good representations, nearest neighbours in feature space should be semantically similar images — same object, same scene, same concept — rather than just visually similar (same colour palette, similar pixel textures).

The quality test: good embeddings produce neighbours that are class-coherent and semantically meaningful even when pixel-level similarity is low. Bad embeddings produce neighbours that look similar as images (same dominant colour, similar composition) but belong to different classes or concepts. This distinction tells you whether the network is truly recognising content or just matching surface statistics.

### 3.2  t-SNE Visualisation

t-SNE (t-Distributed Stochastic Neighbour Embedding) reduces high-dimensional feature vectors to 2D while preserving local neighbourhood structure. Unlike PCA (which is a linear projection and loses non-linear structure), t-SNE can reveal clusters and manifold structure in the learned representations. For CNN features, a good t-SNE plot shows tight clusters per class with clear separation between classes and meaningful layout — related classes (e.g., different dog breeds) cluster near each other; unrelated ones (dog vs aircraft carrier) are distant.

```python
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = ImageFolder('data', transform=preprocess)   # one sub-folder per class
loader  = DataLoader(dataset, batch_size=32, shuffle=False)
print('Classes:', dataset.classes, '| number of images:', len(dataset))

# ── Extract embeddings from the second FC layer (4096-d) ──────────────
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# VGG16 classifier: [Linear(25088,4096), ReLU, Dropout,
#                    Linear(4096,4096),  ReLU, Dropout,
#                    Linear(4096,1000)]
# We want the 4096-d output of the second FC layer (after its ReLU)
embedding_model = nn.Sequential(
    model.features,
    model.avgpool,
    nn.Flatten(),
    model.classifier[:5],   # up to and including the second ReLU
).to(device).eval()

chunks = []
with torch.no_grad():
    for images, _ in loader:
        chunks.append(embedding_model(images.to(device)).cpu())
embeddings = torch.cat(chunks).numpy()              # (N, 4096)
labels     = np.array(dataset.targets)              # (N,)
print(f'Embeddings shape: {embeddings.shape}')

# ── k-NN in feature space ─────────────────────────────────────────────
knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddings)
dists, idxs = knn.kneighbors(embeddings)
neighbours  = idxs[:, 1:]                           # drop each image itself (nearest neighbour #0)

same_class = (labels[neighbours] == labels[:, None]).mean()
print(f'Fraction of 5-NN neighbours with the same class as the query: {same_class:.1%}')

# Show one query image and its 5 nearest neighbours
q = int(np.random.default_rng(0).integers(len(dataset)))
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, i, title in zip(axes, [q] + list(neighbours[q]), ['query'] + [f'NN {k}' for k in range(1, 6)]):
    ax.imshow(Image.open(dataset.samples[i][0]).convert('RGB'))
    ax.set_title(f'{title}: {dataset.classes[labels[i]]}')
    ax.axis('off')
plt.show()

# ── t-SNE dimensionality reduction ────────────────────────────────────
# Step 1: reduce to 50D with PCA (t-SNE is slow on raw 4096-d data)
pca     = PCA(n_components=min(50, len(embeddings) - 1))
emb_pca = pca.fit_transform(embeddings)
print(f'PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}')

# Step 2: t-SNE to 2D (perplexity must be smaller than the number of images)
tsne   = TSNE(n_components=2, perplexity=min(30, len(embeddings) // 4), random_state=0)
emb_2d = tsne.fit_transform(emb_pca)                # (N, 2)

plt.figure(figsize=(6, 6))
for cls, name in enumerate(dataset.classes):
    mask = labels == cls
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, s=12)
plt.legend(); plt.title('t-SNE of VGG16 FC7 features')
plt.show()
# Good result: tight clusters per class, with the classes clearly separated
```

*Code 2 – Extracting 4096-d FC embeddings from VGG16, running k-NN in feature space, and projecting to 2D with PCA + t-SNE. PCA first reduces to 50D (fast, captures most variance) before t-SNE, which is the standard efficiency trick for high-dimensional inputs.*

---

## 4  Saliency Maps and Grad-CAM

### 4.1  The Question: Which Pixels Matter?

A saliency map answers: for a given image and a given predicted class, which pixels had the most influence on that prediction? This is important both for debugging (is the model looking at the right thing?) and for scientific understanding (what does the model consider discriminative?).

The dumbbell example from the slides is instructive: a VGG network trained on ImageNet produced class-maximising images for 'dumbbell' that always included muscular arms. The model had learned that dumbbells and arms always co-occur in training images, and was using both to make its prediction — a perfectly rational strategy given the training data, but not the behaviour we intended. Saliency maps make these co-adaptations visible.

### 4.2  Saliency via Occlusion

The conceptually simplest approach: systematically mask out a patch of the image, run a forward pass, and record how much the predicted class probability drops. Repeat for every position in the image to produce a 2D heat map. Large drops indicate that the masked region was important for the prediction; small drops indicate it was irrelevant.

The main weakness is speed: for a $224 \times 224$ image with a $16 \times 16$ occlusion patch and stride 8, you need roughly $(224/8)^2 = 784$ forward passes. For a large network this can take minutes per image. Gradient-based methods (Section 4.3) produce the same information in a single forward+backward pass.

### 4.3  Gradient-Based Saliency

The gradient of the class score with respect to the input pixels tells us how sensitive the prediction is to each pixel — the saliency map:

$$\text{Saliency} = \left|\frac{\partial y_c}{\partial x_{ij}}\right|$$

where $y_c$ is the unnormalised score for class $c$ and $x_{ij}$ is the intensity of pixel $(i, j)$. Large gradient magnitude at pixel $(i, j)$ means that a small change at that pixel would significantly change the class score — the network is paying attention to that pixel. This requires one forward pass and one backward pass, regardless of image size.

### 4.4  Grad-CAM: Class Activation Maps

Grad-CAM (Selvaraju et al., 2017) produces a coarser but more class-discriminative saliency map by working with the last convolutional layer rather than the input pixels. The intuition: the last conv layer contains the highest-level spatial information before the global pooling and FC layers destroy spatial resolution. The gradient of the class score with respect to each channel of the last conv layer tells us how important that channel is for the predicted class.

The three-step computation:

- **Step 1 — Compute channel importance weights**: $\partial y_c / \partial A^k$, where $A^k$ is the $k$-th channel of the last feature map. Global-average-pool these gradients over the spatial dimensions to get a scalar importance weight $w^c_k$ for each channel $k$.
- **Step 2 — Weighted combination of feature maps**: $L^c_\text{Grad-CAM} = \text{ReLU}\!\left(\sum_k w^c_k \cdot A^k\right)$. Sum all channels weighted by their importance. The ReLU discards channels that negatively influence the class score — we only want to see what supports the prediction, not what suppresses it.
- **Step 3 — Upsample to input resolution**: The last feature map for VGG16 is $14 \times 14$. Bilinear upsampling to $224 \times 224$ gives the final class-discriminative heat map, overlaid on the original image.

Grad-CAM localises the relevant object region well (it knows 'dog' comes from the left half of the image) but lacks fine-grained detail. Guided Grad-CAM fuses it with a pixel-level gradient saliency to recover both the correct region and the fine-grained discriminative features.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, size=224, normalise=True):
    """Load an image as a (1, 3, size, size) tensor, preprocessed for ImageNet models."""
    tfms = [T.Resize(size), T.CenterCrop(size), T.ToTensor()]
    if normalise:
        tfms.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(tfms)(Image.open(path).convert('RGB')).unsqueeze(0)

def denormalise(x):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1)

# ════════════════════════════════════════════════════════════════════
# GRAD-CAM: Class Activation Mapping via gradients
# Works with any CNN that has a convolutional layer before pooling.
# ════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.feature_maps = None   # stores forward activations
        self.gradients    = None   # stores backward gradients

        # Forward hook: capture the feature maps of the target layer
        target_layer.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()           # (1, C, H, W)
        # Ask autograd to hand us d(score)/d(output) during the backward pass
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad.detach()                # (1, C, H, W)

    def generate(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)     # forward pass: triggers _save_features

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backpropagate the raw logit of the target class (not the softmax probability)
        self.model.zero_grad()
        logits[0, class_idx].backward()   # fills self.gradients

        # Step 1: global-average-pool the gradients over the spatial dims
        # weights: how important is each channel for this class?
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)       # (1, C, 1, 1)

        # Step 2: weighted sum of the feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # Step 3: ReLU: keep only regions that support the class
        cam = F.relu(cam)

        # Step 4: upsample to the input image size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze()                                           # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ── Set up VGG16 ──────────────────────────────────────────────────────
weights_vgg = models.VGG16_Weights.IMAGENET1K_V1
categories  = weights_vgg.meta['categories']
model = models.vgg16(weights=weights_vgg).eval()
for m in model.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False            # hooks and in-place ReLUs do not mix

# Target: the ReLU after the last conv layer (features[29] in VGG16)
grad_cam = GradCAM(model, target_layer=model.features[29])

# ── One image, two different classes ──────────────────────────────────
x = load_image('data/cat_and_dog.jpg')

with torch.no_grad():
    logits = model(x)[0]
cat_idx = 281 + logits[281:286].argmax().item()    # best-scoring cat class (281-285)
dog_idx = 151 + logits[151:269].argmax().item()    # best-scoring dog class (151-268)
top_idx = logits.argmax().item()                   # the model's overall prediction
print('Top prediction:', categories[top_idx])

original = denormalise(x)[0].permute(1, 2, 0)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(original); axes[0].set_title('Input')
for ax, idx in zip(axes[1:], [cat_idx, dog_idx]):
    cam, _ = grad_cam.generate(x, class_idx=idx)
    ax.imshow(original); ax.imshow(cam.numpy(), alpha=0.5, cmap='jet')
    ax.set_title(f'Grad-CAM for: {categories[idx]}')
for ax in axes: ax.axis('off')
plt.show()
# Expected: the "cat" map lights up the cat, the "dog" map lights up the dog.
# Same image, same network: only the class whose score we backpropagate differs.

# ── Simple gradient saliency map (pixel-level), for comparison ────────
x2 = x.clone().requires_grad_(True)
logits2 = model(x2)
cls2    = logits2.argmax(1).item()
model.zero_grad()
logits2[0, cls2].backward()

saliency = x2.grad.abs().max(dim=1)[0]            # max over colour channels: (1, 224, 224)

plt.figure(figsize=(5, 5))
plt.imshow(saliency[0].numpy(), cmap='gray'); plt.axis('off')
plt.title(f'Gradient saliency: {categories[cls2]}')
plt.show()
```

*Code 3 – Grad-CAM from scratch using a forward hook on the target layer and a tensor hook that captures its gradients. The four-step pipeline maps exactly to the equations: global-average-pool gradients (Step 1), weighted sum of feature maps (Step 2), ReLU (Step 3), bilinear upsampling (Step 4). The pixel saliency map at the bottom is even simpler: just `x.grad.abs()` after backpropagating the class score.*

---

## 5  Reconstruction-Based Visualisation

All techniques in this section share the same structure: fix the network weights, define a loss based on internal activations, and optimise the input image using gradient ascent or descent. The network is used purely as a differentiable function to evaluate whether the current image matches the desired activation pattern. This is gradient descent on the image rather than on the weights.

### 5.1  Inverting ConvNets

Mahendran & Vedaldi (2015) asked: given the feature representation of an image at layer $l$, can we reconstruct the original image? The answer reveals what information is retained and what is discarded at each layer. The procedure initialises a generated image $\mathbf{y}$ with random noise, then minimises the difference between $\mathbf{y}$'s layer-$l$ feature maps and those of the original image $\mathbf{x}$:

$$\mathcal{L}_\text{content} = \|\mathbf{a}^l(\mathbf{y}) - \mathbf{a}^l(\mathbf{x})\|_F^2 \qquad \text{(Frobenius norm)}$$

Gradients flow back through the network (whose weights are frozen) to $\mathbf{y}$, updating its pixels to make the feature maps match. As $l$ increases (deeper layers), the reconstruction loses fine-grained details but preserves high-level semantic content — confirming that deeper layers are more invariant and abstract. Five reconstructions from the same features look quite different but are equivalent from the network's perspective because the network cannot distinguish between them.

### 5.2  Filter Visualisation via Gradient Ascent

To understand what pattern a specific convolutional filter has learned to detect, we find the input image that maximally activates that filter:

- Initialise $\mathbf{y}$ with random noise (small-amplitude, to avoid NaN from large initial activations).
- Forward pass: compute the mean activation of the target filter's output — this is the loss to maximise.
- Backward pass: compute gradients of that mean activation with respect to $\mathbf{y}$.
- Update $\mathbf{y}$ by taking a small step in the gradient direction (gradient ascent).
- Apply regularisation after each step (clip pixel values, apply Gaussian blur) to keep the image in the range of natural images.

The resulting image is the 'preferred stimulus' of that filter — the pattern it is maximally sensitive to. Early-layer filters produce Gabor-like edge patterns and colour gradients. Mid-layer filters produce repeating textures. Late-layer filters produce recognisable object parts.

```python
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, size=224, normalise=True):
    tfms = [T.Resize(size), T.CenterCrop(size), T.ToTensor()]
    if normalise:
        tfms.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(tfms)(Image.open(path).convert('RGB')).unsqueeze(0)

def denormalise(x):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1)

def frozen_front(model, layer_idx):
    """The first layers of VGG up to and including model.features[layer_idx], frozen."""
    feat = model.features[:layer_idx + 1].to(device).eval()
    for p in feat.parameters():
        p.requires_grad_(False)       # freeze the network
    return feat

# ── Content reconstruction: invert a layer's activations ─────────────
def content_reconstruction(model, content_img, layer_idx, n_steps=300, lr=0.05):
    """
    Find an image whose activations at layer_idx match those of content_img.
    model:       pretrained VGG
    content_img: (1, 3, H, W) tensor (the reference image)
    layer_idx:   which VGG features layer to match
    """
    feat = frozen_front(model, layer_idx)
    content_img = content_img.to(device)

    # Target activations of the content image
    with torch.no_grad():
        target = feat(content_img)

    # Initialise the generated image with small random noise
    generated = (torch.randn_like(content_img) * 0.01).requires_grad_(True)
    optimiser = torch.optim.Adam([generated], lr=lr)

    for step in range(n_steps):
        optimiser.zero_grad()
        loss = F.mse_loss(feat(generated), target)   # match activations
        loss.backward()                               # gradients flow to the pixels
        optimiser.step()

        with torch.no_grad():
            generated.clamp_(-2.5, 2.5)               # keep pixels in a valid range

        if step % 100 == 0:
            print(f'  layer {layer_idx:2d}  step {step:4d}  loss: {loss.item():.4f}')

    return generated.detach().cpu()

# ── Filter visualisation via gradient ascent ──────────────────────────
def visualise_filter(model, layer_idx, filter_idx, n_steps=150, lr=0.1):
    """
    Find the input image that maximally activates a specific filter.
    Gradient ASCENT on the mean filter activation.
    """
    feat = frozen_front(model, layer_idx)

    # Start from small random noise
    img = (torch.randn(1, 3, 224, 224, device=device) * 0.01).requires_grad_(True)
    optimiser = torch.optim.Adam([img], lr=lr)

    for step in range(n_steps):
        optimiser.zero_grad()
        # Loss = NEGATIVE mean activation of the target filter
        # (negative because the optimiser minimises, but we want to maximise)
        loss = -feat(img)[0, filter_idx].mean()
        loss.backward()
        optimiser.step()

        # Regularise: keep the image natural-looking
        with torch.no_grad():
            img.clamp_(-2.5, 2.5)
            if step % 10 == 9:                       # small blur suppresses high-frequency noise
                img.copy_(TF.gaussian_blur(img, kernel_size=3, sigma=0.5))

    return img.detach().cpu()

def to_display(x):
    """Scale an optimised image to [0, 1] for plt.imshow."""
    x = x[0]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x.permute(1, 2, 0)

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# ── 1) Reconstruct a real image from different layers ─────────────────
content = load_image('data/Cat03.jpg')
layers  = {3: 'relu1_2', 8: 'relu2_2', 15: 'relu3_3', 22: 'relu4_3', 29: 'relu5_3'}

fig, axes = plt.subplots(1, 1 + len(layers), figsize=(3 * (1 + len(layers)), 3.4))
axes[0].imshow(denormalise(content)[0].permute(1, 2, 0)); axes[0].set_title('Original')
for ax, (idx, name) in zip(axes[1:], layers.items()):
    recon = content_reconstruction(model, content, layer_idx=idx)
    ax.imshow(denormalise(recon)[0].permute(1, 2, 0)); ax.set_title(f'from {name}')
for ax in axes: ax.axis('off')
plt.show()
# Expected: early layers reproduce the image almost exactly; deeper layers keep the
# overall layout/semantics but lose fine detail and colour precision.

# ── 2) Visualise what individual filters respond to ───────────────────
# Layer indices are conv layers: 0 = conv1_1, 14 = conv3_3, 28 = conv5_3
filters_to_show = {0: [0, 1, 2, 3], 14: [0, 1, 2, 3], 28: [0, 1, 2, 3]}
fig, axes = plt.subplots(len(filters_to_show), 4, figsize=(10, 7.5))
for r, (layer_idx, fids) in enumerate(filters_to_show.items()):
    for c, f in enumerate(fids):
        vis = visualise_filter(model, layer_idx=layer_idx, filter_idx=f)
        axes[r, c].imshow(to_display(vis)); axes[r, c].axis('off')
        axes[r, c].set_title(f'features[{layer_idx}] filter {f}')
plt.show()
# Expected: early filters: oriented edges / colour blobs; middle: textures; late: object-part-like patterns
```

*Code 4 – Content reconstruction and filter visualisation. Both use the same pattern: frozen network, gradient flow to the input image, Adam optimiser on pixels. Content reconstruction minimises MSE to match a target activation. Filter visualisation maximises mean activation via gradient ascent (hence the negative sign in the loss).*

---

## 6  DeepDream

DeepDream (Google, 2015) is a creative extension of filter visualisation: instead of starting from random noise to maximise a specific filter, start from a real image and ask the network to amplify whatever it already detects. The result is a hallucinogenic image where the network's features are exaggerated and made visible.

The procedure: choose a layer (lower layers produce textures and brushstroke patterns; higher layers produce faces, animals, and complex objects). Forward pass the image. Compute the L2 norm of all activations at that layer as the loss to maximise. Backpropagate to the image pixels, add the gradients directly to the image, and repeat. Each iteration amplifies whatever the network detects — a cloud that looks slightly like a bird becomes more bird-like, which makes the network detect the bird more strongly on the next pass, and so on.

The choice of layer controls the visual character of the dream. Layer 1 produces simple edge and colour patterns. Layer 3 produces textured surfaces. Layers 4–5 produce eyes, faces, and animal parts emerging from arbitrary textures — the network's learned templates imposing themselves onto whatever it finds in the image.


---

## 7  Texture Synthesis and the Gram Matrix

### 7.1  What Is the Gram Matrix?

The Gram matrix of a layer's feature maps captures the statistical co-occurrence of features. If the feature map at layer $l$ has shape $(C, H, W)$, we reshape it to a $(C, H \times W)$ matrix and compute the $C \times C$ inner product matrix:

$$\mathbf{G}^l = \mathbf{F}^l (\mathbf{F}^l)^T \qquad \text{where } \mathbf{F}^l \text{ has shape } (C, H \times W)$$

$$G^l_{kk'} = \sum_{ij} a^l_{ijk} \cdot a^l_{ijk'}$$

The entry $G^l_{kk'}$ measures how much filter $k$ and filter $k'$ co-activate at the same spatial location. Large $G^l_{kk'}$: filters $k$ and $k'$ tend to fire together — their patterns co-occur in the image, which is a key characteristic of texture. Near-zero $G^l_{kk'}$: these features appear independently and rarely overlap.

The Gram matrix discards spatial information (where features occur) while preserving statistical information (what features co-occur). This is precisely what we want for texture: a texture is defined by the pattern of feature co-occurrences, not their specific spatial arrangement. Two images of the same brick wall texture have different spatial layouts of individual bricks but the same Gram matrix statistics.

### 7.2  Texture Synthesis Algorithm

Given a reference texture image, synthesise a new texture by finding an image with matching Gram matrices across multiple layers. The loss is the sum of Frobenius distances between Gram matrices:

$$\mathcal{L}_\text{style} = \sum_l \lambda_l \cdot \|\mathbf{G}^l(\mathbf{y}) - \mathbf{G}^l(\mathbf{x})\|_F^2$$

Starting from white noise and minimising $\mathcal{L}_\text{style}$ via gradient descent on $\mathbf{y}$ produces a new image with the same statistical texture properties as $\mathbf{x}$ but different spatial arrangement. The choice of which layers to include controls the scale of texture features matched: lower layers capture fine-grained textures (individual pixels, brush strokes); higher layers capture coarser patterns (repeated motifs, structural elements).

---

## 8  Neural Style Transfer

### 8.1  Combining Content and Style

Neural Style Transfer (Gatys et al., 2015) combines the content reconstruction and texture synthesis ideas into a single optimisation. Given a content image $C$ and a style image $S$, find a generated image $\mathbf{y}$ that simultaneously:

- Matches the feature map activations of $C$ at a chosen content layer (preserving semantic content).
- Matches the Gram matrices of $S$ across multiple style layers (transferring artistic style/texture).

The total loss is a weighted combination:

$$\mathcal{L}_\text{total} = \alpha \cdot \mathcal{L}_\text{content}(\mathbf{y}, C) + \beta \cdot \mathcal{L}_\text{style}(\mathbf{y}, S)$$

The ratio $\alpha/\beta$ controls the trade-off: high $\alpha/\beta$ preserves more content but less style; low $\alpha/\beta$ transfers more style at the expense of content fidelity. Starting $\mathbf{y}$ from the content image (rather than random noise) typically gives better convergence and more coherent results. The key practical tip from the paper: replace max pooling with average pooling for style transfer — average pooling gives smoother gradient flow, producing more visually pleasing results.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image_01(path, size=256):
    """Load an image as a (1, 3, size, size) tensor with pixels in [0, 1]."""
    tfm = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return tfm(Image.open(path).convert('RGB')).unsqueeze(0)

# ════════════════════════════════════════════════════════════════════
# NEURAL STYLE TRANSFER — full implementation
# Content + Style loss via a frozen VGG19 network.
# ════════════════════════════════════════════════════════════════════

def gram_matrix(feature_map):
    """
    Compute the Gram matrix of a feature map.
    feature_map: (1, C, H, W)
    Returns:     (C, C) matrix of feature co-occurrences
    """
    B, C, H, W = feature_map.shape
    feats = feature_map.reshape(C, H * W)           # (C, H*W)
    # Gram matrix = feats @ feats^T, normalised by the number of elements
    return feats @ feats.t() / (C * H * W)

class StyleTransfer:
    def __init__(self,
                 content_layers=['features.22'],           # relu4_2 in VGG19
                 style_layers=['features.1', 'features.6',   # relu1_1, relu2_1
                               'features.11', 'features.20', # relu3_1, relu4_1
                               'features.29']):             # relu5_1
        self.content_layers = content_layers
        self.style_layers   = style_layers
        self.last_layer     = max(int(k.split('.')[1]) for k in content_layers + style_layers)

        # Load VGG19 and replace max pooling by average pooling (smoother gradients)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, nn.MaxPool2d):
                vgg.features[i] = nn.AvgPool2d(2, 2)
            elif isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.model = vgg.features.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)   # freeze all weights

        # VGG expects ImageNet-normalised input; we keep our images in [0, 1] and normalise here
        self.normalise = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_features(self, img):
        """Run a forward pass and collect activations at the chosen layers."""
        features = {}
        x = self.normalise(img)
        for i, layer in enumerate(self.model):
            x = layer(x)
            key = f'features.{i}'
            if key in self.content_layers or key in self.style_layers:
                features[key] = x
            if i == self.last_layer:
                break                  # no need to run the deeper layers
        return features

    def run(self, content_img, style_img, n_steps=300,
            content_weight=1.0, style_weight=1e6):
        content_img, style_img = content_img.to(device), style_img.to(device)

        # Pre-compute target features (only once — images are fixed)
        content_feats = self.get_features(content_img)
        style_feats   = self.get_features(style_img)
        style_grams   = {k: gram_matrix(v) for k, v in style_feats.items()
                         if k in self.style_layers}

        # Initialise the generated image from the content image (better convergence than noise)
        generated = content_img.clone().requires_grad_(True)
        optimiser = torch.optim.LBFGS([generated], max_iter=20)

        step = [0]
        def closure():
            generated.data.clamp_(0, 1)
            optimiser.zero_grad()
            gen_feats = self.get_features(generated)

            # Content loss: MSE between feature maps at the content layer
            L_content = sum(
                F.mse_loss(gen_feats[l], content_feats[l])
                for l in self.content_layers
            )

            # Style loss: MSE between Gram matrices at the style layers
            L_style = sum(
                F.mse_loss(gram_matrix(gen_feats[l]), style_grams[l])
                for l in self.style_layers
            )

            loss = content_weight * L_content + style_weight * L_style
            loss.backward()

            step[0] += 1
            if step[0] % 50 == 0:
                print(f'Step {step[0]:4d}  content: {L_content.item():.2f}',
                      f' style: {L_style.item():.4f}')
            return loss

        for _ in range(n_steps // 20 + 1):
            optimiser.step(closure)

        return generated.detach().clamp(0, 1).cpu()

# ── Usage: a photo of Paris in the style of Starry Night ──────────────
content = load_image_01('data/paris.jpg')
style   = load_image_01('data/starry_night.jpg')

nst    = StyleTransfer()
result = nst.run(content, style, n_steps=300, content_weight=1.0, style_weight=1e6)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, im, title in zip(axes, [content, style, result], ['Content', 'Style', 'Result']):
    ax.imshow(im[0].permute(1, 2, 0)); ax.set_title(title); ax.axis('off')
plt.show()

print('Content weight high: preserves photo, adds subtle style')
print('Style weight high:   strong artistic style, loses photo details')
print('Content layer: relu4_2 (middle layers work best)')
print('Style layers:  relu1_1 through relu5_1 (all scales of texture)')
```

*Code 5 – Complete Neural Style Transfer. The `gram_matrix` function is the core of the style loss — 4 lines implementing the $C \times C$ feature co-occurrence matrix. The key design choices: average pooling instead of max pooling (smoother gradients, better visuals), initialise from content image (better convergence), LBFGS optimiser (converges much faster than Adam for NST).*

---

## 9  Adversarial Attacks

The same gradient-through-frozen-network pattern that enables DeepDream and style transfer also enables adversarial attacks: small perturbations to an image that are imperceptible to humans but cause a CNN to misclassify with high confidence. The technique is directly analogous to class-based reconstruction (Section 5), except the perturbation is constrained to be small:

- Start from a correctly classified image.
- Choose a target misclassification class (can be arbitrary).
- Backpropagate the target class score to the input pixels.
- Take a small step in the gradient direction — the Fast Gradient Sign Method (FGSM) uses $\text{sign}(\partial y_\text{target}/\partial \mathbf{x})$ scaled by a small $\varepsilon$.
- The perturbed image looks identical to the original but the network now classifies it as the target class with near-certainty.

Adversarial examples reveal that CNNs have not learned the same invariances as human perception. The perturbations exploit directions in pixel space that are highly influential for the network but imperceptible to the human visual system. They are also transferable: an adversarial example crafted to fool one network often fools other networks trained on the same data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path, size=224, normalise=True):
    tfms = [T.Resize(size), T.CenterCrop(size), T.ToTensor()]
    if normalise:
        tfms.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(tfms)(Image.open(path).convert('RGB')).unsqueeze(0)

# ── Fast Gradient Sign Method (FGSM) adversarial attack ──────────────
def fgsm_attack(model, image, true_label, epsilon=0.03):
    """
    Untargeted attack: make the model misclassify the image.
    image:   (1, 3, H, W) with pixel values in [0, 1]
    epsilon: perturbation magnitude (small = imperceptible to humans)
    """
    model.eval()
    img = image.clone().requires_grad_(True)

    # Forward pass
    logits = model(img)
    loss   = F.cross_entropy(logits, true_label)

    # Backpropagate to the input image (not to model weights)
    model.zero_grad()
    loss.backward()

    # FGSM: perturb in the direction that increases the loss
    # sign() makes the perturbation exactly +/- epsilon per pixel
    perturbation    = epsilon * img.grad.sign()
    adversarial_img = (img + perturbation).clamp(0, 1).detach()

    return adversarial_img

def targeted_attack(model, image, target_class, epsilon=0.03, n_steps=50):
    """
    Targeted attack: make the model output a specific class.
    Uses iterative gradient ascent on the target class score.
    """
    model.eval()
    img = image.clone().requires_grad_(True)
    opt = torch.optim.Adam([img], lr=epsilon)

    for step in range(n_steps):
        opt.zero_grad()
        logits = model(img)
        # Maximise the target class score (gradient ascent → negate for Adam)
        loss = -logits[0, target_class]
        loss.backward()
        opt.step()

        # Project back: keep perturbation small and pixels valid
        with torch.no_grad():
            delta = img - image
            delta.clamp_(-epsilon, epsilon)  # L-inf constraint
            img.data = (image + delta).clamp(0, 1)

    return img.detach()

# ── Test ──────────────────────────────────────────────────────────────
weights = models.ResNet50_Weights.IMAGENET1K_V2
categories = weights.meta['categories']

# Normalisation is part of the model, so the attack works directly on pixels in [0, 1]
model = nn.Sequential(T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                      models.resnet50(weights=weights)).eval()
for p in model.parameters():
    p.requires_grad_(False)

image = load_image('data/Cat03.jpg', normalise=False)   # real image, pixels in [0, 1]

def predict(x):
    probs = F.softmax(model(x), dim=1)
    conf, idx = probs.max(dim=1)
    return idx.item(), conf.item()

# Attack the model's own prediction on the clean image
clean_idx, clean_conf = predict(image)
print(f'Clean image:       {categories[clean_idx]} ({clean_conf:.1%})')

# FGSM attack
eps       = 0.03
adv_img   = fgsm_attack(model, image, torch.tensor([clean_idx]), epsilon=eps)
perturbation_magnitude = (adv_img - image).abs().max().item()
print(f'Max pixel perturbation: {perturbation_magnitude:.4f} (eps={eps}) — barely visible')

adv_idx, adv_conf = predict(adv_img)
print(f'Adversarial image: {categories[adv_idx]} ({adv_conf:.1%})')

# Targeted attack: make the model say "goldfish" (ImageNet class 1)
target_class = 1
tgt_img = targeted_attack(model, image, target_class, epsilon=eps)
tgt_idx, tgt_conf = predict(tgt_img)
print(f'Targeted attack:   {categories[tgt_idx]} ({tgt_conf:.1%})   [target was: {categories[target_class]}]')

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
panels = [(image, f'Clean: {categories[clean_idx]}'),
          (adv_img, f'FGSM: {categories[adv_idx]}'),
          ((adv_img - image) / (2 * eps) + 0.5, 'FGSM perturbation (amplified)'),
          (tgt_img, f'Targeted: {categories[tgt_idx]}')]
for ax, (im, title) in zip(axes, panels):
    ax.imshow(im[0].clamp(0, 1).permute(1, 2, 0)); ax.set_title(title); ax.axis('off')
plt.show()
```

*Code 6 – FGSM adversarial attack. The key line is `epsilon * img.grad.sign()`: the sign function makes the perturbation exactly $\varepsilon$ in magnitude per pixel, giving the attacker maximum influence within a fixed budget. This shows that the same backpropagation machinery that trains networks can be turned against them.*

---

## 10  Summary: The Gradient-as-Visualisation Toolkit

Every technique in this lecture is a variation on one theme: treat the input image as the variable to optimise, differentiate a scalar loss through a frozen network, and interpret the result. The choice of loss function determines what question you are asking:

| Technique | Loss function | Optimise | What it reveals |
|---|---|---|---|
| Layer activations | N/A (just forward pass) | Nothing | What each filter detected |
| Filter visualisation | Negative mean filter activation | Input image | What pattern maximally excites a filter |
| Content reconstruction | MSE of layer activations | Input image | What info is retained at each layer |
| Gradient saliency | Class score $y_c$ | Nothing (just grad) | Which pixels most affect prediction |
| Grad-CAM | Class score $y_c$ | Nothing (just grad) | Which image regions are discriminative |
| DeepDream | Negative mean squared activation | Input image | What the network already sees amplified |
| Texture synthesis | MSE of Gram matrices | Input image | What texture looks like to the network |
| Neural Style Transfer | Content + style losses | Input image | Artistic style blended with content |
| Adversarial attack | Target class score | Input image | Network vulnerabilities / brittleness |

The practical take-home: these techniques should be used routinely, not just out of curiosity. Grad-CAM on validation set failures often reveals in minutes what would take hours of debugging to find otherwise — the model looking at the wrong region, a spurious background correlation, a class boundary that doesn't match human intuition. t-SNE of the penultimate layer tells you whether the network's learned embedding space has the cluster structure you expect before you commit to training a downstream task. Filter visualisation tells you whether training has converged cleanly. Together, these tools turn the black box into something you can inspect, debug, and reason about.

---

## 11  Exercises
The exercises are representative of the types of questions you might encounter at the written exam (except that the exam questions will be formulated as multiple choice). You are welcome to work on the exercises during TØ on Thursdays.

### Exercise 1 — Why Visualise ConvNets?

**(a)** A classifier distinguishing huskies from wolves reaches 95% test accuracy, but a visualisation shows it mostly looks at the background of the image. Explain what has probably gone wrong, and why high test accuracy alone did not reveal it.

**(b)** Explain why a neural network is not invertible, i.e. why you cannot simply "read off" what the network was shown from its output. An example is fine.

**(c)** Because of (b), every visualisation technique in this lecture is an approximation. Describe the one operation that almost all of them have in common (what is fixed, what is differentiated, and with respect to what).

---

### Exercise 2 — What the Layers Learn, and Layer Activations

```python
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

activations = {}
def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

model.features[0].register_forward_hook(make_hook('conv1_1'))
model.features[28].register_forward_hook(make_hook('conv5_3'))

with torch.no_grad():
    _ = model(img)

for name, act in activations.items():
    frac_dead = (act == 0).float().mean().item()
    print(f'{name}: shape={tuple(act.shape)}  dead={frac_dead:.1%}')
```

**(a)** What does the forward hook do?

**(b)** Describe in overall terms what the feature maps of early layers, middle layers, and late layers typically look like when plotted for a given image.

**(c)** Many feature maps in the last convolutional layer (`conv5_3`) are entirely black (all zeros) for a given image. Is this a sign that something is wrong? Explain.

**(d)** The same measurement is used during training as a diagnostic. If a ReLU layer shows all-zero feature maps for many different inputs, what does this indicate, and what is a likely cause?

---

### Exercise 3 — Visualising Filters

**(a)** Why can the filters of the *first* convolutional layer be displayed directly as colour images, while the filters of deeper layers cannot?

**(b)** What should the first-layer filters of a well-trained network look like? What do noisy, grainy filters tell you about the training?

**(c)** How would you go about visualising what a filter in a deeper layer responds to, given that it cannot be shown directly? (Name the technique; Exercise 7 asks you to describe it in detail.)

---

### Exercise 4 — Inspecting FC Layers: k-NN and t-SNE

```python
embedding_model = nn.Sequential(
    model.features,
    model.avgpool,
    nn.Flatten(),
    model.classifier[:5],   # up to and including the second ReLU
)

with torch.no_grad():
    embeddings = embedding_model(images)      # (N, 4096)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(embeddings.numpy())
dists, idxs = knn.kneighbors(embeddings[[0]].numpy())

pca     = PCA(n_components=50)
emb_pca = pca.fit_transform(embeddings.numpy())
tsne    = TSNE(n_components=2, perplexity=30)
emb_2d  = tsne.fit_transform(emb_pca)
```

**(a)** What does the `embedding_model` produce for each image, and which layer of VGG16 does it stop at?

**(b)** The five nearest neighbours of a query image are found in this embedding space. If the network has learned good representations, what kind of similarity should the neighbours show? What kind of similarity would indicate a *bad* embedding?

**(c)** State what t-SNE aims to achieve (you do not need to explain how it works internally), and what a "good" t-SNE plot of CNN features looks like.

**(d)** Why does the code reduce the data to 50 dimensions with PCA before running t-SNE?

---

### Exercise 5 — Saliency by Occlusion and by Gradient

**(a)** Describe how an occlusion saliency map is computed for one image and one predicted class. What does a large drop in class probability at a given position mean?

**(b)** Occlusion needs one forward pass for every patch position. If you double the stride of the sliding patch (in both directions) while keeping everything else fixed, roughly how does the number of forward passes change? (Answer with a factor; no exact count is needed.) What do you give up?

**(c)** In the lecture's "dumbbell" example, class-maximising images of the dumbbell class always contained muscular arms. What does this reveal about what the model has learned, and why is it a rational outcome given the training data?

---

### Exercise 6 — Grad-CAM

```python
def generate(self, x, class_idx=None):
    self.model.eval()
    logits = self.model(x)
    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()

    self.model.zero_grad()
    logits[0, class_idx].backward()

    weights = self.gradients.mean(dim=[2, 3], keepdim=True)          # (1, C, 1, 1)
    cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)     # (1, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    return cam
```

**(a)** What does `weights` contain, and what does a large value for a given channel mean?

**(b)** Why is the weighted sum of feature maps passed through a ReLU?

**(c)** The backward pass starts from `logits[0, class_idx]`, the raw class score, rather than from the softmax probability. State what this choice means for which class the resulting map explains.

**(d)** The Grad-CAM map for VGG16 is computed on the last convolutional feature map and then upsampled. Explain why it localises the object well but lacks fine detail, and what Guided Grad-CAM adds.

**(e)** Why is Grad-CAM applied to the last convolutional layer rather than to an FC layer?

---

### Exercise 7 — Reconstruction-Based Visualisation

```python
generated = torch.randn_like(content_img) * 0.01
generated.requires_grad_(True)
optimiser = torch.optim.Adam([generated], lr=lr)

for step in range(n_steps):
    optimiser.zero_grad()
    _ = model(generated)
    loss = F.mse_loss(gen_acts['a'], target)
    loss.backward()
    optimiser.step()
```

**(a)** What are the parameters being optimised here, and what has been done to the network's own weights? Explain how the usual "train a network" setup has been turned around.

**(b)** What does `target` contain, and what does the loss measure? Which image is the "content image"?

**(c)** Reconstructions from the features of an early layer look very different from reconstructions from a late layer. Describe the difference, and what it says about what deeper layers retain and discard.

**(d)** Filter visualisation uses the same machinery with a different loss: the *negative* mean activation of one filter. Explain why the loss is negative, and what the resulting image shows.

**(e)** Why do filter-visualisation implementations clip pixel values or apply a small blur after each step?

---

### Exercise 8 — Class-Based Reconstruction and Adversarial Attacks

```python
def fgsm_attack(model, image, true_label, epsilon=0.03):
    img = image.clone().requires_grad_(True)
    logits = model(img)
    loss   = F.cross_entropy(logits, true_label)
    model.zero_grad()
    loss.backward()
    perturbation    = epsilon * img.grad.sign()
    adversarial_img = (img + perturbation).clamp(0, 1).detach()
    return adversarial_img
```

**(a)** Describe class-based reconstruction: what is optimised, and what does the resulting image show?

**(b)** What is an adversarial attack? What do the perturbed and original image look like to a human, and to the network?

**(c)** In `fgsm_attack`, why is `.sign()` applied to the gradient, and what does `epsilon` control?

**(d)** How is this attack related to class-based reconstruction, and what extra constraint makes it an *attack* rather than a visualisation?

---

### Exercise 9 — Texture Synthesis and the Gram Matrix

```python
def gram_matrix(feature_map):
    B, C, H, W = feature_map.shape
    F = feature_map.view(C, H * W)
    return F @ F.t() / (C * H * W)
```

**(a)** What is the shape of the Gram matrix for a feature map with $C$ channels and spatial size $H \times W$? What does the entry in row $k$ and column $k'$ express?

**(b)** The Gram matrix throws away *where* features occur. Explain why this is exactly what a description of a texture needs. Would two photographs of the same brick wall, with the bricks laid out in different positions, have similar or different Gram matrices?

**(c)** Describe how a new texture is synthesised from a reference texture image using Gram matrices (what is the starting point, what is optimised, and what is the loss?).

**(d)** The loss uses Gram matrices from several layers. What does the choice of layers control?

**(e)** What is meant by the "style image"?

---

### Exercise 10 — Neural Style Transfer

```python
L_content = sum(F.mse_loss(gen_feats[l], content_feats[l])
                for l in self.content_layers)
L_style   = sum(F.mse_loss(gram_matrix(gen_feats[l]), style_grams[l])
                for l in self.style_layers)
loss = content_weight * L_content + style_weight * L_style
```

**(a)** What is optimised in this code, what is kept frozen, and which images does the algorithm take as input?

**(b)** Which of the two loss terms makes the result look like the photograph, and which makes it look like the painting? Why is the content loss computed on feature maps from a middle-to-deep layer and not directly on pixels?

**(c)** Why are the style layers chosen from many depths, from the first to the fifth convolutional block?

**(d)** What happens to the output if `style_weight` is made much larger relative to `content_weight`, and what if it is made much smaller?

**(e)** Name two practical choices that help style transfer converge or look better (e.g. what the generated image is initialised with, or what replaces max pooling in the VGG network), and say why each helps.

---

## References

- Zeiler, M. & Fergus, R. (2013). Visualizing and Understanding Convolutional Networks (ZFNet). ECCV 2014.
- Mahendran, A. & Vedaldi, A. (2015). Understanding Deep Image Representations by Inverting Them. CVPR.
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
- Gatys, L. et al. (2015). A Neural Algorithm of Artistic Style. arXiv 1508.06576.
- Gatys, L. et al. (2015). Texture Synthesis Using Deep Convolutional Neural Networks. NeurIPS.
- Goodfellow, I. et al. (2015). Explaining and Harnessing Adversarial Examples. ICLR.
- Google DeepDream blog: ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
- CS231n Stanford: cs231n.github.io/understanding-cnn/