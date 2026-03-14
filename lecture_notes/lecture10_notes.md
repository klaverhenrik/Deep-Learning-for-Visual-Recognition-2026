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

# ── Visualising layer activations with forward hooks ──────────────────
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# Storage for captured activations
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on the first and last conv layers
model.features[0].register_forward_hook(make_hook('conv1_1'))   # 64 filters, 3 channels
model.features[28].register_forward_hook(make_hook('conv5_3'))  # 512 filters

# Run a forward pass
img = torch.randn(1, 3, 224, 224)   # use a real image in practice

with torch.no_grad():
    _ = model(img)

# Inspect the captured activations
for name, act in activations.items():
    n_channels = act.shape[1]
    frac_dead  = (act == 0).float().mean().item()
    print(f'{name}: shape={tuple(act.shape)}  dead={frac_dead:.1%}')
# conv1_1: shape=(1,64,224,224)   dead=~10%  (low — most filters fire early)
# conv5_3: shape=(1,512,14,14)    dead=~60%  (high — specialised filters)

# ── Visualising first-layer filters directly ──────────────────────────
# First conv layer weights have shape (out_ch, in_ch, H, W) = (64, 3, 3, 3)
conv1_weights = model.features[0].weight.data.cpu()  # (64, 3, 3, 3)

# Normalise each filter to [0,1] for visualisation
def normalise_filter(w):
    w = w - w.min()
    w = w / (w.max() + 1e-8)
    return w

for i in range(min(8, conv1_weights.shape[0])):
    filt = normalise_filter(conv1_weights[i])  # (3, 3, 3) — RGB
    # Convert to HWC for display: filt.permute(1,2,0).numpy()
    # Plot with matplotlib: plt.imshow(filt.permute(1,2,0))
    pass

print(f'First layer filters: {conv1_weights.shape}')  # (64, 3, 3, 3)
print('Each filter is a 3x3 RGB image — directly visualisable as colour patches')
print('Healthy filters: smooth, structured, no noise')
```

*Code 1 – Capturing and inspecting layer activations using forward hooks. The `frac_dead` metric immediately flags dead ReLU problems. First-layer filters can be directly rendered as colour images ($3 \times 3$ RGB); deeper filters require the reconstruction-based approach in Section 5.*

---

## 3  Inspecting FC Layers: k-NN and t-SNE

### 3.1  Nearest Neighbours in Feature Space

Fully connected layers produce high-dimensional embeddings (e.g. 4096-d in AlexNet/VGGNet) that summarise an image's semantic content. To inspect what the network has learned, extract these embeddings for a large image collection and search for nearest neighbours using k-NN. If the network has learned good representations, nearest neighbours in feature space should be semantically similar images — same object, same scene, same concept — rather than just visually similar (same colour palette, similar pixel textures).

The quality test: good embeddings produce neighbours that are class-coherent and semantically meaningful even when pixel-level similarity is low. Bad embeddings produce neighbours that look similar as images (same dominant colour, similar composition) but belong to different classes or concepts. This distinction tells you whether the network is truly recognising content or just matching surface statistics.

### 3.2  t-SNE Visualisation

t-SNE (t-Distributed Stochastic Neighbour Embedding) reduces high-dimensional feature vectors to 2D while preserving local neighbourhood structure. Unlike PCA (which is a linear projection and loses non-linear structure), t-SNE can reveal clusters and manifold structure in the learned representations. For CNN features, a good t-SNE plot shows tight clusters per class with clear separation between classes and meaningful layout — related classes (e.g., different dog breeds) cluster near each other; unrelated ones (dog vs aircraft carrier) are distant.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ── Extract embeddings from penultimate FC layer ──────────────────────
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Remove the final classification layer to get 4096-d embeddings
# VGG16 classifier: [Linear(25088,4096), ReLU, Dropout,
#                    Linear(4096,4096),  ReLU, Dropout,
#                    Linear(4096,1000)]
# We want the 4096-d output of the second FC layer (after ReLU)
embedding_model = nn.Sequential(
    model.features,
    model.avgpool,
    nn.Flatten(),
    model.classifier[:5],   # up to and including second ReLU
)
embedding_model.eval()

# Simulated dataset: 200 images, 10 classes
N, C = 200, 10
images = torch.randn(N, 3, 224, 224)
labels = torch.randint(0, C, (N,))

with torch.no_grad():
    embeddings = embedding_model(images)   # (N, 4096)
print(f'Embeddings shape: {embeddings.shape}')   # (200, 4096)

# ── k-NN in feature space ─────────────────────────────────────────────
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(embeddings.numpy())

# Find 5 nearest neighbours of the first image
query   = embeddings[[0]].numpy()
dists, idxs = knn.kneighbors(query)
print('Nearest neighbour labels (should match query label for good embeddings):')
print(f'Query label: {labels[0].item()}')
print(f'Neighbour labels: {labels[idxs[0]].tolist()}')

# ── t-SNE dimensionality reduction ────────────────────────────────────
# Step 1: Reduce to 50D with PCA (t-SNE is slow on raw 4096-d data)
pca       = PCA(n_components=50)
emb_pca   = pca.fit_transform(embeddings.numpy())
print(f'PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}')

# Step 2: t-SNE to 2D
tsne    = TSNE(n_components=2, perplexity=30, random_state=0, n_iter=1000)
emb_2d  = tsne.fit_transform(emb_pca)   # (N, 2)
print(f'2D embedding shape: {emb_2d.shape}')  # (200, 2)

# Plot: colour each point by its class label
# import matplotlib.pyplot as plt
# for cls in range(C):
#     mask = labels.numpy() == cls
#     plt.scatter(emb_2d[mask,0], emb_2d[mask,1], label=str(cls), s=10)
# plt.legend(); plt.title('t-SNE of VGG16 penultimate layer features')
# Good result: tight clusters per class, semantically related classes nearby
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
        # Backward hook: capture the gradients flowing back through that layer
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()   # (1, C, H, W)

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()  # (1, C, H, W)

    def generate(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)     # forward pass — triggers _save_features

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backpropagate the score of the target class
        # (not the softmax probability — the raw logit)
        self.model.zero_grad()
        logits[0, class_idx].backward()   # triggers _save_gradients

        # Step 1: Global-average-pool the gradients over spatial dims
        # weights: how important is each channel for this class?
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Step 2: Weighted sum of feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # Step 3: ReLU — keep only channels that support the prediction
        cam = F.relu(cam)

        # Step 4: Upsample to input image size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear',
                            align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze()  # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ── Apply Grad-CAM to VGG16 ───────────────────────────────────────────
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Target the last conv layer (features[28] in VGG16)
grad_cam = GradCAM(model, target_layer=model.features[28])

x = torch.randn(1, 3, 224, 224, requires_grad=True)
cam, predicted_class = grad_cam.generate(x)

print(f'Grad-CAM map shape: {cam.shape}')       # (224, 224)
print(f'Predicted class:    {predicted_class}')
print(f'CAM range:          [{cam.min():.3f}, {cam.max():.3f}]')   # [0, 1]

# To visualise: overlay cam as a heatmap on the original image
# import matplotlib.pyplot as plt
# plt.imshow(original_image)
# plt.imshow(cam.numpy(), alpha=0.5, cmap='jet')
# plt.title(f'Grad-CAM — class {predicted_class}')

# ── Simple gradient saliency map (pixel-level) ───────────────────────
model.eval()
x2 = torch.randn(1, 3, 224, 224, requires_grad=True)
logits2 = model(x2)
cls2    = logits2.argmax(1).item()
model.zero_grad()
logits2[0, cls2].backward()

# Saliency = absolute gradient with respect to input pixels
saliency = x2.grad.abs()          # (1, 3, 224, 224)
saliency = saliency.max(dim=1)[0] # max over colour channels: (1, 224, 224)
print(f'Pixel saliency shape: {saliency.shape}')  # (1, 224, 224)
# High values = pixels that most influence the prediction
```

*Code 3 – Grad-CAM from scratch using forward and backward hooks. The four-step pipeline maps exactly to the equations: global-average-pool gradients (Step 1), weighted sum of feature maps (Step 2), ReLU (Step 3), bilinear upsampling (Step 4). The pixel saliency map at the bottom is even simpler: just `x.grad.abs()` after backpropagating the class score.*

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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ── Content reconstruction: invert a layer's activations ─────────────
def content_reconstruction(model, content_img, layer_idx, n_steps=500, lr=0.05):
    """
    Find an image whose activations at layer_idx match content_img.
    model:       frozen VGG-style network
    content_img: (1, 3, H, W) tensor (the reference image)
    layer_idx:   which VGG features layer to match
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # freeze the network

    # Extract the target activations from the content image
    target_acts = {}
    def hook(m, i, o): target_acts['a'] = o
    handle = model.features[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        _ = model(content_img)
    target = target_acts['a'].clone()
    handle.remove()

    # Initialise the generated image with small random noise
    generated = torch.randn_like(content_img) * 0.01
    generated.requires_grad_(True)
    optimiser = torch.optim.Adam([generated], lr=lr)

    gen_acts = {}
    handle2 = model.features[layer_idx].register_forward_hook(
        lambda m,i,o: gen_acts.update({'a': o}))

    for step in range(n_steps):
        optimiser.zero_grad()
        _ = model(generated)           # forward pass through frozen network
        loss = F.mse_loss(gen_acts['a'], target)  # match activations
        loss.backward()                # gradients flow to generated pixels
        optimiser.step()

        # Keep pixels in valid range
        with torch.no_grad():
            generated.clamp_(-2.5, 2.5)

        if step % 100 == 0:
            print(f'Step {step:4d}  loss: {loss.item():.4f}')

    handle2.remove()
    return generated.detach()

# ── Filter visualisation via gradient ascent ──────────────────────────
def visualise_filter(model, layer_idx, filter_idx, n_steps=300, lr=0.1):
    """
    Find the input image that maximally activates a specific filter.
    Gradient ASCENT on the mean filter activation.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Start from small random noise
    img = torch.randn(1, 3, 224, 224) * 0.01
    img.requires_grad_(True)
    optimiser = torch.optim.Adam([img], lr=lr)

    current_act = {}
    handle = model.features[layer_idx].register_forward_hook(
        lambda m,i,o: current_act.update({'a': o}))

    for step in range(n_steps):
        optimiser.zero_grad()
        _ = model(img)

        # Loss = NEGATIVE mean activation of target filter
        # (negative because optimiser minimises, but we want to maximise)
        loss = -current_act['a'][0, filter_idx].mean()
        loss.backward()
        optimiser.step()

        # Regularise: keep image natural-looking
        with torch.no_grad():
            img.clamp_(-2.5, 2.5)
            # Optional: apply small Gaussian blur to suppress high-freq noise
            # img.data = gaussian_blur(img.data, sigma=0.5)

    handle.remove()
    return img.detach()

# ── Quick test ────────────────────────────────────────────────────────
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Visualise filter 0 of layer conv1_1 (first conv layer)
# Expected: simple oriented edge or colour blob
vis_img = visualise_filter(model, layer_idx=0, filter_idx=0, n_steps=200)
print(f'Filter visualisation shape: {vis_img.shape}')  # (1, 3, 224, 224)
print('Early-layer filters: simple edges/colours')
print('Tip: visualise many filters and check they are diverse and structured')
```

*Code 4 – Content reconstruction and filter visualisation. Both use the same pattern: frozen network, gradient flow to the input image, Adam optimiser on pixels. Content reconstruction minimises MSE to match a target activation. Filter visualisation maximises mean activation via gradient ascent (hence the negative sign in the loss).*

---

## 6  DeepDream

DeepDream (Google, 2015) is a creative extension of filter visualisation: instead of starting from random noise to maximise a specific filter, start from a real image and ask the network to amplify whatever it already detects. The result is a hallucinogenic image where the network's features are exaggerated and made visible.

The procedure: choose a layer (lower layers produce textures and brushstroke patterns; higher layers produce faces, animals, and complex objects). Forward pass the image. Compute the L2 norm of all activations at that layer as the loss to maximise. Backpropagate to the image pixels, add the gradients directly to the image, and repeat. Each iteration amplifies whatever the network detects — a cloud that looks slightly like a bird becomes more bird-like, which makes the network detect the bird more strongly on the next pass, and so on.

The choice of layer controls the visual character of the dream. Layer 1 produces simple edge and colour patterns. Layer 3 produces textured surfaces. Layers 4–5 produce eyes, faces, and animal parts emerging from arbitrary textures — the network's learned templates imposing themselves onto whatever it finds in the image.

```python
import torch
import torch.nn as nn
import torchvision.models as models

def deep_dream(model, img, layer_idx, n_steps=100, lr=0.01, octave_scale=1.4,
               n_octaves=4):
    """
    DeepDream: amplify what the network already sees in an image.

    Multi-octave version: apply at multiple scales for richer results.
    model:       frozen CNN
    img:         (1, 3, H, W) starting image
    layer_idx:   which layer to 'dream' with
    lr:          step size
    n_octaves:   number of scale levels (more = richer features)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    current_acts = {}
    handle = model.features[layer_idx].register_forward_hook(
        lambda m,i,o: current_acts.update({'a': o}))

    # Multi-octave: process at multiple resolutions
    octaves = [img]
    for _ in range(n_octaves - 1):
        h, w = octaves[-1].shape[2:]
        new_h, new_w = int(h / octave_scale), int(w / octave_scale)
        octaves.append(torch.nn.functional.interpolate(
            octaves[-1], size=(new_h, new_w), mode='bilinear'))

    detail = torch.zeros_like(octaves[-1])

    for octave in reversed(octaves):
        # Upsample detail to current octave size
        if detail.shape != octave.shape:
            detail = torch.nn.functional.interpolate(
                detail, size=octave.shape[2:], mode='bilinear')
        # Add detail to current octave
        dreamed = (octave + detail).requires_grad_(True)

        # Gradient ascent at this octave
        for step in range(n_steps):
            dreamed.grad = None
            _ = model(dreamed)
            # Loss = mean squared activation (maximise it)
            loss = current_acts['a'].pow(2).mean()
            loss.backward()

            # Normalise gradient for stable steps
            grad = dreamed.grad / (dreamed.grad.abs().mean() + 1e-8)
            with torch.no_grad():
                dreamed += lr * grad
                dreamed.clamp_(-2.5, 2.5)

        detail = (dreamed - octave).detach()

    handle.remove()
    return dreamed.detach()

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
img   = torch.randn(1, 3, 224, 224) * 0.5   # start from a real image in practice

# Layer 4 (conv2_1 in VGG16): produces swirling textures
# Layer 20 (conv4_2):         produces eyes, faces, object parts
dreamed = deep_dream(model, img, layer_idx=4, n_steps=50, n_octaves=2)
print(f'Dream image shape: {dreamed.shape}')  # (1, 3, 224, 224)
print('Low layer (4):   textural patterns, brush strokes')
print('High layer (20): eyes, faces, animal parts emerging')
```

*Code 5 – DeepDream with multi-octave processing. The gradient normalisation trick (divide by mean absolute value) keeps steps stable — without it, a few large-gradient pixels dominate and create speckling artefacts. Multi-octave processing applies the dream at multiple scales for richer, more detailed results.*

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
    # Reshape to (C, H*W)
    F = feature_map.view(C, H * W)
    # Gram matrix = F @ F^T, normalised by spatial size
    return F @ F.t() / (C * H * W)

class StyleTransfer:
    def __init__(self,
                 content_layers=['features.21'],        # conv4_2 in VGG19
                 style_layers=['features.0','features.5',  # conv1_1, conv2_1
                               'features.10','features.19', # conv3_1, conv4_1
                               'features.28']):           # conv5_1
        self.content_layers = content_layers
        self.style_layers   = style_layers

        # Load VGG19 with average pooling instead of max pooling
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Replace MaxPool2d with AvgPool2d for smoother gradients
        for name, module in vgg.features.named_children():
            if isinstance(module, nn.MaxPool2d):
                vgg.features[int(name)] = nn.AvgPool2d(2, 2)

        self.model = vgg.features.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)   # freeze all weights

    def get_features(self, img):
        """Run forward pass and collect activations at named layers."""
        features = {}
        x = img
        for name, layer in self.model.named_children():
            x = layer(x)
            key = f'features.{name}'
            if key in self.content_layers or key in self.style_layers:
                features[key] = x
        return features

    def run(self, content_img, style_img, n_steps=300,
            content_weight=1.0, style_weight=1e6):
        # Pre-compute target features (only once — images are fixed)
        content_feats = self.get_features(content_img)
        style_feats   = self.get_features(style_img)
        style_grams   = {k: gram_matrix(v) for k,v in style_feats.items()
                         if k in self.style_layers}

        # Initialise generated image from content (better convergence than noise)
        generated = content_img.clone().requires_grad_(True)
        optimiser = torch.optim.LBFGS([generated], max_iter=20)

        step = [0]
        def closure():
            generated.data.clamp_(0, 1)
            optimiser.zero_grad()
            gen_feats = self.get_features(generated)

            # Content loss: MSE between feature maps at content layer
            L_content = sum(
                F.mse_loss(gen_feats[l], content_feats[l])
                for l in self.content_layers
            )

            # Style loss: MSE between Gram matrices at style layers
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

        return generated.detach().clamp(0, 1)

# ── Usage ─────────────────────────────────────────────────────────────
nst = StyleTransfer()
content = torch.rand(1, 3, 224, 224)  # your content image here
style   = torch.rand(1, 3, 224, 224)  # your style image here

# result = nst.run(content, style, n_steps=300,
#                  content_weight=1.0, style_weight=1e6)

print('Content weight high: preserves photo, adds subtle style')
print('Style weight high:   strong artistic style, loses photo details')
print('Content layer: conv4_2 (middle layers work best)')
print('Style layers:  conv1_1 through conv5_1 (all scales of texture)')
```

*Code 6 – Complete Neural Style Transfer. The `gram_matrix` function is the core of the style loss — 4 lines implementing the $C \times C$ feature co-occurrence matrix. The key design choices: average pooling instead of max pooling (smoother gradients, better visuals), initialise from content image (better convergence), LBFGS optimiser (converges much faster than Adam for NST).*

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
import torch.nn.functional as F
import torchvision.models as models

# ── Fast Gradient Sign Method (FGSM) adversarial attack ──────────────
def fgsm_attack(model, image, true_label, epsilon=0.03):
    """
    Untargeted attack: make the model misclassify the image.
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
    # sign() keeps the perturbation within [-epsilon, +epsilon] per pixel
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
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

image     = torch.rand(1, 3, 224, 224)   # replace with a real image
true_lbl  = torch.tensor([207])           # some true label

# FGSM attack
adv_img   = fgsm_attack(model, image, true_lbl, epsilon=0.03)
perturbation_magnitude = (adv_img - image).abs().max().item()
print(f'Max pixel perturbation: {perturbation_magnitude:.4f}',
      f'(ε={0.03}) — imperceptible to humans')

with torch.no_grad():
    orig_pred = model(image).argmax(1).item()
    adv_pred  = model(adv_img).argmax(1).item()
print(f'Original prediction:    {orig_pred}')
print(f'Adversarial prediction: {adv_pred}')  # likely different!
```

*Code 7 – FGSM adversarial attack. The key line is `epsilon * img.grad.sign()`: the sign function makes the perturbation exactly $\varepsilon$ in magnitude per pixel, giving the attacker maximum influence within a fixed budget. This shows that the same backpropagation machinery that trains networks can be turned against them.*

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

## References

- Zeiler, M. & Fergus, R. (2013). Visualizing and Understanding Convolutional Networks (ZFNet). ECCV 2014.
- Mahendran, A. & Vedaldi, A. (2015). Understanding Deep Image Representations by Inverting Them. CVPR.
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
- Gatys, L. et al. (2015). A Neural Algorithm of Artistic Style. arXiv 1508.06576.
- Gatys, L. et al. (2015). Texture Synthesis Using Deep Convolutional Neural Networks. NeurIPS.
- Goodfellow, I. et al. (2015). Explaining and Harnessing Adversarial Examples. ICLR.
- Google DeepDream blog: ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
- CS231n Stanford: cs231n.github.io/understanding-cnn/