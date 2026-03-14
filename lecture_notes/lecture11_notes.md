# Lecture 11 — Sequence Models

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes build sequence modelling from first principles: language models and word embeddings, vanilla RNNs, the vanishing gradient problem that motivates LSTMs, the bottleneck problem that motivates attention, seq2seq with attention, and the step from LSTM-based attention to the Transformer. Image captioning ties computer vision and NLP together as the closing application.

---

## 1  Sequence Modelling

### 1.1  Why Sequences?

A vast number of real-world problems involve ordered data. Language is the most prominent example — the meaning of a word depends on its context, and meaning depends on order. Other sequence domains include audio, video, time-series, and DNA.

A plain MLP applied to individual tokens cannot handle sequences: it sees each token independently, with no knowledge of what came before or after.

### 1.2  Paradigms

| Paradigm | Input | Output | Example |
|---|---|---|---|
| Many-to-one | Sequence | Single label | Sentiment analysis |
| Many-to-many (same length) | Sequence | Label per token | POS tagging, NER |
| Many-to-many (different length) | Sequence | Sequence | Machine translation |
| One-to-many | Single input | Sequence | Image captioning |

### 1.3  Language Models

A language model assigns a probability to a sequence of words via the chain rule:

$$P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, \ldots, x_{i-1})$$

A language model is equivalent to a **next-word predictor**. Training is self-supervised — the data itself provides labels.

```python
import torch
import torch.nn as nn

# Word embeddings: learnable lookup table
vocab_size, embed_dim = 10000, 128
embedding = nn.Embedding(vocab_size, embed_dim)

token_ids = torch.tensor([42, 1337, 7, 88])   # token IDs for ['He','saw','two','birds']
vecs = embedding(token_ids)   # (4, 128)
print(f'Embedding shape: {vecs.shape}')
```

---

## 2  Vanilla RNNs

### 2.1  The Hidden State

An RNN maintains a hidden state $\mathbf{h}_t$ that summarises everything seen so far:

$$\mathbf{h}_t = f_\text{RNN}(\mathbf{h}_{t-1}, \mathbf{x}_t), \qquad \hat{\mathbf{y}}_t = f_\text{MLP}(\mathbf{h}_t)$$

The vanilla RNN computes:

$$\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1} + \mathbf{V}\mathbf{x}_t + \mathbf{b})$$

The same parameters $\mathbf{W}, \mathbf{V}, \mathbf{b}$ are applied at every time step — parameter sharing across time.

### 2.2  The Vanishing Gradient Problem

Training via backpropagation through time multiplies many Jacobians together. Each includes the derivative of tanh, which is at most 0.25. Multiplying many such terms causes gradients to shrink exponentially as they flow back through early time steps. After 20–50 steps the gradient is effectively zero. This is the same vanishing gradient problem as in deep feedforward networks, but appearing along the time dimension.

---

## 3  Long Short-Term Memory (LSTM)

### 3.1  The Cell State: A Gradient Highway

The LSTM (Hochreiter & Schmidhuber, 1997) adds a **cell state** $\mathbf{c}_t$ that provides a direct gradient path through time. The cell state is modified only by additive updates, so gradients can flow backwards without exponential attenuation.

Three **gates** act as differentiable switches:

### 3.2  The Six Equations

$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \qquad \text{Forget gate}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \qquad \text{Input gate}$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \qquad \text{Candidate cell}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \qquad \text{Cell state update}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \qquad \text{Output gate}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \qquad \text{Hidden state}$$

The **cell state update** $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ is the critical line. The addition (not multiplication by weight matrices) provides the gradient highway.

> **LSTMs and ResNets: the same insight.** Both solve the vanishing gradient problem with an additive skip path. In ResNets: $h(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$ (spatial skip). In LSTMs: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ (temporal skip). The ResNet authors were aware of the LSTM literature.

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([h_prev, x], dim=1)
        f, i, c_tilde, o = self.linear(combined).chunk(4, dim=1)

        f       = torch.sigmoid(f)       # forget gate
        i       = torch.sigmoid(i)       # input gate
        c_tilde = torch.tanh(c_tilde)    # candidate
        o       = torch.sigmoid(o)       # output gate

        c_new = f * c_prev + i * c_tilde  # additive update = gradient highway
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

# PyTorch built-in
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
x = torch.randn(4, 8, 32)   # batch=4, seq_len=8, input=32
out, (h_n, c_n) = lstm(x)
print(f'Output: {out.shape}')    # (4, 8, 64)
print(f'Final h: {h_n.shape}')  # (2, 4, 64) — 2 layers
```

---

## 4  Sequence-to-Sequence and the Bottleneck

The seq2seq architecture (encoder + decoder) handles variable-length output from variable-length input.

**Encoder**: reads the full input sequence and compresses it into a single context vector $\mathbf{h}_\text{encoder}$ (the final hidden state).

**Decoder**: generates output autoregressively, conditioned on $\mathbf{h}_\text{encoder}$:

$$P(Y \mid X) = \prod_{i=1}^{|Y|} P(y_i \mid X, y_1, \ldots, y_{i-1})$$

**The bottleneck problem**: the entire input — regardless of length — must be compressed into one fixed-size vector. Information is inevitably lost. As sequence length grows, the bottleneck becomes catastrophic.

> *"You can't cram the meaning of a whole %&!$ing sentence into a single $&!*ing vector!"* — Ray Mooney

---

## 5  Attention Mechanism

### 5.1  The Core Idea

Rather than working from a single context vector, the decoder dynamically **looks back at all encoder hidden states** and weights them by relevance to the current decoding step.

The computation at each decoder step:

$$a_i = \mathbf{q} \cdot \mathbf{k}_i \qquad \text{(alignment scores)}$$

$$\alpha_i = \text{softmax}(\mathbf{a})_i \qquad \text{(attention weights)}$$

$$\mathbf{c} = \sum_i \alpha_i \mathbf{v}_i \qquad \text{(context vector)}$$

where $\mathbf{q}$ is the query (from decoder), $\mathbf{k}_i$ and $\mathbf{v}_i$ are keys and values (from encoder).

### 5.2  Self-Attention

When queries, keys, and values all come from the same sequence, this is **self-attention**. It updates each token's embedding to reflect its relationships with all other tokens — solving the context-dependence problem ('Apple' the fruit vs 'Apple' the company).

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores  = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, V), weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.embed   = nn.Embedding(vocab, embed)
        self.encoder = nn.LSTM(embed, hidden, batch_first=True)
        self.decoder = nn.LSTM(embed + hidden, hidden, batch_first=True)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, src, tgt):
        enc_out, (h, c) = self.encoder(self.embed(src))
        all_logits = []
        for t in range(tgt.size(1)):
            emb   = self.embed(tgt[:, t:t+1])
            q     = self.W_q(h.transpose(0,1))
            ctx, _ = attention(q, self.W_k(enc_out), self.W_v(enc_out))
            inp   = torch.cat([emb, ctx], dim=-1)
            dec, (h, c) = self.decoder(inp, (h, c))
            all_logits.append(self.out(dec))
        return torch.cat(all_logits, dim=1)   # (B, tgt_len, vocab)

model = Seq2SeqWithAttention(1000, 64, 128)
src   = torch.randint(0, 1000, (2, 5))
tgt   = torch.randint(0, 1000, (2, 4))
print(model(src, tgt).shape)   # (2, 4, 1000)
```

---

## 6  From Seq2Seq to Transformers

**The sequential bottleneck of RNNs**: computing $\mathbf{h}_5$ requires $\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3, \mathbf{h}_4$ first. This serial dependency prevents parallelisation, severely limiting training speed on modern GPUs.

**The Transformer insight**: if self-attention already lets every token communicate with every other token, we don't need the RNN at all. Removing the LSTM and keeping only attention makes all tokens computable in parallel.

| Property | Vanilla RNN | LSTM | Transformer |
|---|---|---|---|
| Long-range dependencies | Poor (vanishing grad) | Good (cell state) | Excellent (direct attention) |
| Parallelism | Sequential only | Sequential only | Fully parallel |
| Memory cost | $O(1)$ per step | $O(1)$ per step | $O(n^2)$ attention matrix |
| Training speed | Slow | Moderate | Fast on GPU/TPU |

---

## 7  Image Captioning

Image captioning is the unifying application: one image in, a sequence of words out.

### 7.1  CNN-LSTM with Spatial Attention

The classical approach extracts spatial feature maps from a CNN (shape $H \times W \times D$, one feature vector per spatial location) and applies attention over these spatial features at each decoding step. When generating the word 'hat', the attention weights should highlight the hat region. This provides both better captions and interpretable visualisations of what the model attended to at each word.

```python
import torch, torch.nn as nn
import torchvision.models as models

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, feat_dim=2048):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])   # → (B,2048,7,7)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.embed   = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, vocab_size)

    def encode_image(self, img):
        f = self.cnn(img)                          # (B, 2048, 7, 7)
        B, C, H, W = f.shape
        return self.feat_proj(f.permute(0,2,3,1).reshape(B, H*W, C))  # (B, 49, H)

    def forward(self, images, captions):
        vf = self.encode_image(images)             # (B, 49, H)
        h  = vf.mean(1);  c = torch.zeros_like(h)
        logits_all, attn_all = [], []
        for t in range(captions.size(1)):
            q = self.W_q(h.unsqueeze(1))
            w = torch.softmax((q * self.W_k(vf)).sum(-1), dim=-1)   # (B, 49)
            ctx = (w.unsqueeze(-1) * self.W_v(vf)).sum(1)
            h, c = self.decoder(torch.cat([self.embed(captions[:,t]), ctx], 1), (h,c))
            logits_all.append(self.output(h).unsqueeze(1))
            attn_all.append(w.unsqueeze(1))
        return torch.cat(logits_all,1), torch.cat(attn_all,1)

cap = ImageCaptioner(5000)
imgs = torch.randn(2, 3, 224, 224)
capts = torch.randint(0, 5000, (2, 10))
logits, attn = cap(imgs, capts)
print(f'Logits: {logits.shape}')  # (2, 10, 5000)
print(f'Attn maps: {attn.shape}') # (2, 10, 49) — reshape [:,:,t] to (7,7) for viz
```

---

## 8  Language Model Families

| Model | Architecture | Pre-training task | Use cases |
|---|---|---|---|
| BERT | Encoder only | Masked token prediction (bidirectional) | Classification, NER, QA |
| GPT | Decoder only | Next token prediction (autoregressive) | Text generation, in-context learning |
| BART/T5 | Encoder-decoder | Denoising | Translation, summarisation |

**GPT-3 in-context learning**: with 175B parameters trained purely on next-token prediction, GPT-3 can perform new tasks by reading a few examples in the prompt — without any gradient updates. This works because the diversity of pre-training makes the model learn to learn from examples.

---

## 9  Summary

| Architecture | Key mechanism | Motivated by |
|---|---|---|
| Vanilla RNN | Shared hidden state $\mathbf{h}_t$ | MLPs cannot handle variable-length sequences |
| LSTM | Cell state + gating | Vanilla RNN: vanishing gradient |
| Seq2Seq | Encoder-decoder with fixed context | LSTM: no variable-length output |
| Seq2Seq + Attention | Soft alignment over all encoder states | Seq2Seq: single context vector bottleneck |
| Transformer | Self-attention replaces recurrence | LSTM + attention: sequential = slow |

Two themes connect this lecture to the broader course. First, the **inductive bias trade-off**: RNNs bake sequential processing into their architecture; Transformers make no such assumption and let patterns emerge from data — just as CNNs vs ViTs. Second, the **encoder-decoder pattern**: every architecture here (seq2seq, image captioning, Stable Diffusion's U-Net with cross-attention) follows the same structure — a powerful encoder compresses the input; a task-specific decoder produces the output.

## References

- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Bahdanau, D. et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.
- Xu, K. et al. (2015). Show, Attend and Tell. ICML.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Brown, T. et al. (2020). GPT-3. NeurIPS.
