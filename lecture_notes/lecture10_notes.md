# Lecture 10 — Visualising and Understanding ConvNets

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes cover every visualisation technique from simple filter inspection through Grad-CAM to Neural Style Transfer. The unifying theme: every technique reduces to differentiating a loss with respect to pixels through a frozen network, treating the image as the variable to optimise while the weights stay fixed.

---

## 1  Why Visualise ConvNets?

Neural networks are often described as black boxes. This is practically dangerous: a classifier achieving 95% accuracy might be using background correlations, dataset biases, or spurious co-occurrences rather than the genuine discriminative features.

The famous example: a 'husky vs wolf' classifier achieving high accuracy by detecting snow in the background — most training wolves were photographed in snowy environments.

The challenge: networks are **not invertible**. Many different inputs can produce the same output, so there is no direct way to read off what the network is doing. Every visualisation technique is an indirect answer to a carefully designed question.

> **The unifying pattern**: run a forward pass through a frozen network, define a scalar loss based on some internal activation or output, then differentiate that loss with respect to the input pixels using backpropagation. The gradient map answers a specific interpretability question. Neural Style Transfer, Grad-CAM, filter visualisation, and adversarial attacks are all instances of this pattern.

---

## 2  Layer Activations and Filter Visualisation

### 2.1  Layer Activations

The most direct visualisation: plot the feature maps produced by each layer for a given input. Healthy patterns across layers:

- **Early layers (conv1, conv2)**: dense activations across most of the image; filters respond to oriented edges and colour blobs everywhere
- **Middle layers**: progressively less interpretable; abstract combinations of early features
- **Late layers (conv5)**: many all-zero feature maps — filters looking for complex patterns not present in this particular image

Training diagnostic: for ReLU networks, activations start broad and become more sparse and localised as training progresses. Feature maps that are all-zero for many different inputs indicate **dead ReLUs** — a symptom of too-high learning rate.

### 2.2  First-Layer Filter Visualisation

First-layer filters have exactly 3 channels (RGB) and can be directly visualised as colour images. Well-trained networks show clean, smooth, structured filters. Noisy, grainy filters indicate poor convergence or insufficient regularisation.

```python
import torch
import torch.nn as nn
import torchvision.models as models

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# Capture activations with forward hooks
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        activations[name] = out.detach().cpu()
    return hook

model.features[0].register_forward_hook(make_hook('conv1_1'))
model.features[28].register_forward_hook(make_hook('conv5_3'))

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    _ = model(x)

for name, act in activations.items():
    frac_dead = (act == 0).float().mean().item()
    print(f'{name}: shape={tuple(act.shape)}  dead={frac_dead:.1%}')

# First-layer weights: (64, 3, 3, 3) — directly visualisable as 64 RGB patches
w = model.features[0].weight.data.cpu()   # (64, 3, 3, 3)
print(f'Conv1 weights shape: {w.shape}')
# Normalise each filter to [0,1]: (w - w.min()) / (w.max() - w.min())
```

---

## 3  FC Layer Inspection: k-NN and t-SNE

The penultimate FC layer produces a high-dimensional embedding (4096-d for VGGNet) that summarises each image's semantic content. Two ways to inspect it:

**k-NN**: find the nearest neighbours of a query image in embedding space. Good embeddings produce semantically similar neighbours (same object, same scene) even when pixel-level similarity is low. Bad embeddings produce colour/texture matches that are semantically unrelated.

**t-SNE**: reduce 4096-d embeddings to 2D while preserving local neighbourhood structure. Good representations show tight, well-separated clusters per class with semantically related classes placed nearby.

```python
import torch, torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# Remove final FC layer to get 4096-d embeddings
embedder = nn.Sequential(*list(model.children())[:-1], nn.Flatten(),
                          model.classifier[:5])   # up to second ReLU
embedder.eval()

N, labels = 200, torch.randint(0, 10, (200,))
X = torch.randn(N, 3, 224, 224)
with torch.no_grad():
    emb = embedder(X)   # (200, 4096)

# k-NN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(emb.numpy())
_, idxs = knn.kneighbors(emb[[0]].numpy())
print(f'Query label: {labels[0].item()},  NN labels: {labels[idxs[0]].tolist()}')

# t-SNE (PCA first for speed)
emb_50  = PCA(n_components=50).fit_transform(emb.numpy())
emb_2d  = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(emb_50)
print(f'2D embedding shape: {emb_2d.shape}')  # (200, 2)
```

---

## 4  Grad-CAM: Class Activation Maps

Grad-CAM (Selvaraju et al., 2017) produces a class-discriminative spatial heatmap using the gradient of the class score with respect to the last convolutional layer.

**Step 1** — global-average-pool the gradients over spatial dimensions to get channel importance weights:

$$w_k^c = \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$$

where $A^k$ is the $k$-th channel of the last feature map and $y^c$ is the unnormalised score for class $c$.

**Step 2** — weighted combination of feature maps, with ReLU to keep only positive influence:

$$L^c_\text{Grad-CAM} = \text{ReLU}\!\left(\sum_k w_k^c A^k\right)$$

**Step 3** — bilinear upsample to input resolution.

```python
import torch, torch.nn.functional as F
import torchvision.models as models

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = None
        self.gradients    = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'feature_maps', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Step 1: channel importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Step 2: weighted combination + ReLU
        cam = (weights * self.feature_maps).sum(1, keepdim=True)
        cam = F.relu(cam)

        # Step 3: upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

model    = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
grad_cam = GradCAM(model, target_layer=model.features[28])
x        = torch.randn(1, 3, 224, 224, requires_grad=True)
cam, cls = grad_cam.generate(x)
print(f'Grad-CAM: {cam.shape}, predicted class: {cls}')  # (224, 224)

# Pixel-level gradient saliency (even simpler)
model.eval()
x2 = torch.randn(1, 3, 224, 224, requires_grad=True)
model(x2)[0, model(x2).argmax(1).item()].backward()
saliency = x2.grad.abs().max(dim=1)[0]
print(f'Saliency map: {saliency.shape}')  # (1, 224, 224)
```

---

## 5  Reconstruction-Based Visualisation

### 5.1  Content Reconstruction (Inverting ConvNets)

Find an image $\mathbf{y}$ whose layer-$\ell$ activations match those of a reference image $\mathbf{x}$:

$$J_\text{content} = \| \mathbf{a}^\ell(\mathbf{y}) - \mathbf{a}^\ell(\mathbf{x}) \|_F^2$$

Minimise with respect to $\mathbf{y}$ (frozen network, gradient to pixels). As $\ell$ increases, reconstructions lose fine-grained details but preserve high-level semantic content — confirming that deeper layers are more invariant and abstract.

### 5.2  Filter Visualisation via Gradient Ascent

Find the input that **maximally activates** a specific filter — the pattern it has learned to detect:

$$\max_\mathbf{y} \; \text{mean}\!\left(A^k(\mathbf{y})\right)$$

Initialise from small random noise, gradient ascent, regularise (clip pixel values, optional Gaussian blur) to keep the image in the range of natural images.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

def visualise_filter(model, layer_idx, filter_idx, n_steps=200, lr=0.1):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    img = torch.randn(1, 3, 224, 224) * 0.01
    img.requires_grad_(True)
    opt = torch.optim.Adam([img], lr=lr)

    current = {}
    handle = model.features[layer_idx].register_forward_hook(
        lambda m, i, o: current.update({'a': o}))

    for step in range(n_steps):
        opt.zero_grad()
        _ = model(img)
        loss = -current['a'][0, filter_idx].mean()   # maximise (ascent)
        loss.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(-2.5, 2.5)

    handle.remove()
    return img.detach()

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vis   = visualise_filter(model, layer_idx=0, filter_idx=0, n_steps=100)
print(f'Filter visualisation: {vis.shape}')  # (1, 3, 224, 224)
```

---

## 6  DeepDream

Feed a real image to a frozen network, pick a layer, and **amplify what the network already detects**:

1. Forward pass
2. Set the gradient equal to the activations themselves (maximise their L2 norm)
3. Add the gradient to the image
4. Repeat

Each iteration amplifies whatever the network detects — clouds that look slightly like birds become more bird-like, which makes the network detect birds more strongly, and so on. Lower layers produce textural patterns; higher layers produce recognisable object parts.

```python
def deep_dream(model, img, layer_idx, n_steps=50, lr=0.01):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    acts = {}
    handle = model.features[layer_idx].register_forward_hook(
        lambda m, i, o: acts.update({'a': o}))

    dreamed = img.clone().requires_grad_(True)
    for _ in range(n_steps):
        dreamed.grad = None
        _ = model(dreamed)
        loss = acts['a'].pow(2).mean()    # maximise squared activations
        loss.backward()
        grad = dreamed.grad / (dreamed.grad.abs().mean() + 1e-8)  # normalise
        with torch.no_grad():
            dreamed += lr * grad
            dreamed.clamp_(-2.5, 2.5)

    handle.remove()
    return dreamed.detach()

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
img   = torch.randn(1, 3, 224, 224) * 0.5
dream = deep_dream(model, img, layer_idx=4, n_steps=30)
```

---

## 7  Texture Synthesis and the Gram Matrix

### 7.1  The Gram Matrix

The **Gram matrix** of a feature map captures statistical co-occurrence of features — what makes a texture:

$$G^{[\ell]}_{kk'} = \sum_{i,j} A^{[\ell]}_{ijk} \cdot A^{[\ell]}_{ijk'}$$

Equivalently: $\mathbf{G}^{[\ell]} = \mathbf{F}^{[\ell]} (\mathbf{F}^{[\ell]})^T$ where $\mathbf{F}^{[\ell]}$ is the $(C, H \times W)$ reshape of the feature map.

- Large $G_{kk'}$: features $k$ and $k'$ co-activate → they appear together in the texture
- Near-zero $G_{kk'}$: features rarely overlap

The Gram matrix discards spatial arrangement (where features are) while preserving statistical structure (what features co-occur) — precisely the definition of texture.

### 7.2  Texture Synthesis Loss

Match Gram matrices across multiple layers:

$$J_\text{style} = \sum_\ell \lambda_\ell \|\mathbf{G}^{[\ell]}(\mathbf{y}) - \mathbf{G}^{[\ell]}(\mathbf{x}_\text{style})\|_F^2$$

---

## 8  Neural Style Transfer

Combine content reconstruction (Section 5.1) and texture synthesis (Section 7.2):

$$J_\text{total} = \alpha \cdot J_\text{content}(\mathbf{y}, \mathbf{x}_\text{content}) + \beta \cdot J_\text{style}(\mathbf{y}, \mathbf{x}_\text{style})$$

The ratio $\alpha/\beta$ controls the content-style trade-off. Practical tips: use average pooling instead of max pooling (smoother gradient flow); initialise $\mathbf{y}$ from the content image (faster convergence); use LBFGS optimiser.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

def gram_matrix(feat):
    B, C, H, W = feat.shape
    F = feat.view(C, H*W)
    return F @ F.t() / (C * H * W)

class StyleTransfer:
    CONTENT_LAYERS = ['features.21']           # conv4_2
    STYLE_LAYERS   = ['features.0','features.5','features.10','features.19','features.28']

    def __init__(self):
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Replace MaxPool with AvgPool for smoother gradients
        for i, m in enumerate(vgg.features):
            if isinstance(m, nn.MaxPool2d):
                vgg.features[i] = nn.AvgPool2d(2, 2)
        self.net = vgg.features.eval()
        for p in self.net.parameters(): p.requires_grad_(False)

    def get_features(self, img):
        feats, x = {}, img
        for name, layer in self.net.named_children():
            x = layer(x)
            key = f'features.{name}'
            if key in self.CONTENT_LAYERS + self.STYLE_LAYERS:
                feats[key] = x
        return feats

    def run(self, content, style, n_steps=300, alpha=1.0, beta=1e6):
        cf  = self.get_features(content)
        sf  = self.get_features(style)
        sg  = {k: gram_matrix(v) for k, v in sf.items() if k in self.STYLE_LAYERS}

        gen = content.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([gen], max_iter=20)

        step = [0]
        def closure():
            gen.data.clamp_(0, 1)
            opt.zero_grad()
            gf = self.get_features(gen)
            L_content = sum(F.mse_loss(gf[l], cf[l]) for l in self.CONTENT_LAYERS)
            L_style   = sum(F.mse_loss(gram_matrix(gf[l]), sg[l]) for l in self.STYLE_LAYERS)
            loss = alpha * L_content + beta * L_style
            loss.backward()
            step[0] += 1
            if step[0] % 20 == 0:
                print(f'Step {step[0]}  content={L_content.item():.2f}  style={L_style.item():.4f}')
            return loss

        for _ in range(n_steps//20 + 1):
            opt.step(closure)
        return gen.detach().clamp(0, 1)

nst     = StyleTransfer()
content = torch.rand(1, 3, 224, 224)
style   = torch.rand(1, 3, 224, 224)
# result = nst.run(content, style)
```

---

## 9  Adversarial Attacks

The same gradient-through-frozen-network pattern enables adversarial attacks: imperceptible perturbations that cause misclassification.

**FGSM** (Fast Gradient Sign Method):

$$\mathbf{x}_\text{adv} = \mathbf{x} + \epsilon \cdot \text{sign}\!\left(\nabla_\mathbf{x} J\right)$$

```python
import torch, torch.nn.functional as F
import torchvision.models as models

def fgsm(model, image, true_label, epsilon=0.03):
    model.eval()
    img = image.clone().requires_grad_(True)
    loss = F.cross_entropy(model(img), true_label)
    model.zero_grad()
    loss.backward()
    return (img + epsilon * img.grad.sign()).clamp(0, 1).detach()

model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
image    = torch.rand(1, 3, 224, 224)
label    = torch.tensor([207])
adv_img  = fgsm(model, image, label)

with torch.no_grad():
    orig_pred = model(image).argmax(1).item()
    adv_pred  = model(adv_img).argmax(1).item()
print(f'Original: {orig_pred}  Adversarial: {adv_pred}')
```

---

## 10  Summary

| Technique | Loss function | Optimised | Answers |
|---|---|---|---|
| Layer activations | N/A (forward pass only) | Nothing | What each filter detected |
| Filter visualisation | Negative mean filter activation | Input image | What maximally excites a filter |
| Content reconstruction | $\|\mathbf{a}^\ell(\mathbf{y})-\mathbf{a}^\ell(\mathbf{x})\|^2$ | Input image | What info is retained at each layer |
| Gradient saliency | Class score $y^c$ | Nothing (just grad) | Which pixels most affect prediction |
| Grad-CAM | Class score $y^c$ | Nothing (just grad) | Which image regions are discriminative |
| DeepDream | Negative mean squared activation | Input image | What the network already sees, amplified |
| Texture synthesis | $\|\mathbf{G}(\mathbf{y})-\mathbf{G}(\mathbf{x})\|^2$ | Input image | Texture pattern to the network |
| Neural Style Transfer | Content + style losses | Input image | Artistic style + content blended |
| Adversarial attack | Target class score | Input image | Network vulnerabilities |

These tools should be used routinely during development, not only out of curiosity. Grad-CAM on validation failures often reveals in minutes what would take hours of debugging to find — wrong regions, spurious correlations, unexpected class boundaries.

## References

- Zeiler, M. & Fergus, R. (2013). ZFNet. ECCV 2014.
- Mahendran, A. & Vedaldi, A. (2015). Inverting ConvNets. CVPR.
- Selvaraju, R. et al. (2017). Grad-CAM. ICCV.
- Gatys, L. et al. (2015). Neural Style Transfer. arXiv 1508.06576.
- Goodfellow, I. et al. (2015). Adversarial Examples. ICLR.
