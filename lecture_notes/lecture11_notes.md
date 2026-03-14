# Lecture 11
# Sequence Models

*Deep Learning for Visual Recognition · Aarhus University*

These notes build sequence modelling from first principles: language models and word embeddings, vanilla RNNs, the vanishing gradient problem that motivates LSTMs, the bottleneck problem that motivates attention, seq2seq with attention, and the step from LSTM-based attention to the fully attention-based Transformer. The notes end with image captioning as a unifying application that ties computer vision and NLP together.

---

## 1  Sequence Modelling

### 1.1  Why Sequences?

A vast number of real-world problems involve data that is naturally ordered over time or position. Language is the most prominent example — the meaning of a word depends on its context, and the meaning of a sentence depends on the order of its words. Other sequence domains include audio, video, time-series, DNA, and more.

The key requirement that distinguishes sequence models from standard classifiers: the model must handle variable-length inputs and outputs, and predictions must be context-dependent. A plain MLP applied to individual tokens cannot satisfy either requirement — it sees each token independently, with no knowledge of what came before or after.

### 1.2  Sequence Modelling Paradigms

| Paradigm | Input | Output | Example |
|---|---|---|---|
| Many-to-one | Sequence of tokens | Single label | Sentiment analysis |
| Many-to-many (same length) | Sequence | Label per token | POS tagging, NER |
| Many-to-many (diff. length) | Sequence | Sequence | Machine translation |
| One-to-many | Single input | Sequence | Image captioning |

### 1.3  Language Models

A language model assigns a probability to a sequence of words. The key factorisation (chain rule of probability):

$$P(x_1, x_2, \ldots, x_n) = \prod_i P(x_i \mid x_1, \ldots, x_{i-1})$$

This expresses the probability of a sentence as the product of the probability of each word given all preceding words. A language model is therefore equivalent to a next-word predictor: at each step, estimate the probability distribution over the vocabulary given the context so far. Training is self-supervised — the data itself provides labels (the next word in the sequence).

### 1.4  Word Embeddings

Words are represented as dense vectors called word embeddings. The fundamental idea: words that appear in similar contexts should have similar embeddings. 'Good' and 'great' can substitute for each other in most sentences, so they should be close in embedding space. Embeddings are learned — either standalone (Word2Vec, GloVe) or as part of a larger model — and capture rich semantic and syntactic relationships.

Word embeddings can be treated as a lookup table (an embedding matrix of size $|\text{vocabulary}| \times \text{embedding\_dim}$) that maps each word token to a dense vector. This is the `nn.Embedding` layer in PyTorch.

```python
import torch
import torch.nn as nn

# ── Word embeddings ───────────────────────────────────────────────────
vocab_size = 10000   # number of unique words
embed_dim  = 128     # vector dimension for each word

embedding = nn.Embedding(vocab_size, embed_dim)

# Convert a sentence to a sequence of token IDs
# (in practice: use a tokenizer; here we simulate with random IDs)
sentence = ['He', 'saw', 'two', 'birds']
token_ids = torch.tensor([42, 1337, 7, 88])   # example IDs

# Look up embeddings — each word becomes a 128-d vector
word_vecs = embedding(token_ids)   # (4, 128)
print(f'Sentence length: {len(sentence)} words')
print(f'Embedding shape: {word_vecs.shape}')   # (4, 128)

# Words with similar meaning should have high cosine similarity
# (this only holds after training; here the embeddings are random)
import torch.nn.functional as F
sim = F.cosine_similarity(word_vecs[0:1], word_vecs[1:2])
print(f'Cosine similarity (random init): {sim.item():.3f}')
```

*Code 1 – Word embeddings with `nn.Embedding`. The embedding layer is a learnable lookup table: each integer token ID maps to a row of the weight matrix, producing a dense vector. After training, semantically related words cluster in embedding space.*

---

## 2  Recurrent Neural Networks

### 2.1  The Hidden State

A Recurrent Neural Network (RNN) processes a sequence one token at a time, maintaining a hidden state $\mathbf{h}_t$ that summarises everything seen so far. At each step $t$, the hidden state is updated from the previous state and the current input:

$$\mathbf{h}_t = f_\text{RNN}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

$$\hat{\mathbf{y}}_t = f_\text{MLP}(\mathbf{h}_t)$$

The same function $f_\text{RNN}$ and the same parameters are applied at every timestep — parameter sharing across time, just as convolution shares parameters across space. This means the model's size is independent of sequence length, and it can process sequences of any length. The initial hidden state $\mathbf{h}_0$ is typically set to all zeros.

The vanilla RNN computes $\mathbf{h}_t$ with a single tanh layer:

$$\mathbf{h}_t = \tanh(\mathbf{W} \cdot \mathbf{h}_{t-1} + \mathbf{V} \cdot \mathbf{x}_t + \mathbf{b})$$

where $\mathbf{W}$ is the hidden-to-hidden weight matrix, $\mathbf{V}$ is the input-to-hidden matrix, and $\mathbf{b}$ is a bias. The prediction at each step is a softmax over the vocabulary (or a linear layer for regression tasks).

### 2.2  Training: Backpropagation Through Time

Training an RNN means computing gradients of the total loss (sum of losses at each timestep) with respect to the parameters $\mathbf{W}$, $\mathbf{V}$, and $\mathbf{b}$. Because $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$ which depends on $\mathbf{h}_{t-2}$ and so on, gradients must flow backwards through time — the chain rule is applied across all $T$ timesteps. The gradient of the loss at timestep $T$ with respect to a weight in the first RNN cell requires multiplying $T$ Jacobians together.

### 2.3  The Vanishing Gradient Problem in RNNs

Each Jacobian in the backpropagation-through-time chain includes the derivative of tanh, which is at most 1 and often much smaller. Multiplying many such terms together causes the gradient to shrink exponentially as it flows back from later timesteps to earlier ones. After 20–50 steps the gradient at the first few timesteps is effectively zero — those early words contribute nothing to the parameter update, even if they were semantically important.

This is the same vanishing gradient problem encountered in deep feedforward networks (Lectures 3 and 5), but here it appears along the time dimension rather than the depth dimension. RNNs with tanh activations practically cannot learn dependencies spanning more than about 10–20 tokens. For NLP tasks with long sentences or long-range dependencies (subject–verb agreement across a clause, pronoun coreference), this is a serious limitation.

---

## 3  Long Short-Term Memory (LSTM)

### 3.1  The Key Idea: A Separate Memory Lane

The Long Short-Term Memory network (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient problem with a clever architectural addition: a cell state $\mathbf{c}_t$ that runs alongside the hidden state $\mathbf{h}_t$ and provides a direct gradient highway through time. The cell state is modified only by additive updates (no multiplication by tanh weights along the main path), so gradients can flow backwards through it without exponential shrinkage.

The cell state is controlled by three gates — learned functions that output values between 0 and 1, acting as differentiable switches that decide how much information to keep, forget, or add:

### 3.2  The Four Equations

At each timestep $t$, the LSTM computes:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \qquad \text{Forget gate: how much of } \mathbf{c}_{t-1} \text{ to erase}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \qquad \text{Input gate: how much new info to write}$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \qquad \text{Candidate: what new info to write}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \qquad \text{Cell state update}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \qquad \text{Output gate: what to expose as } \mathbf{h}_t$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \qquad \text{Hidden state output}$$

The cell state update equation is the critical one: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$. The addition (not multiplication) means gradients can flow backward through $\mathbf{c}_t$ without being attenuated by weight matrices. The forget gate $\mathbf{f}_t$ can be close to 1.0 across many steps, letting long-range information pass through unchanged — solving the vanishing gradient problem that afflicted vanilla RNNs.

An intuition for each gate:

- **Forget gate $\mathbf{f}_t$**: 'Should I erase what I remember?' For a language model processing a new sentence, this gate would activate at sentence boundaries to clear the previous context.
- **Input gate $\mathbf{i}_t$**: 'How much should I update my memory with this new input?' For rare but important words (named entities, negations), the input gate learns to write strongly.
- **Output gate $\mathbf{o}_t$**: 'What part of my memory should I expose right now?' The cell state stores everything; the output gate selects what is relevant for the current prediction.

> **LSTMs and ResNets: the same insight.** LSTMs (1997) and ResNets (2015) both solve the same underlying problem with the same solution: a direct path for gradient flow that bypasses multiplicative weight layers. In ResNets the skip connection is spatial ($\mathbf{x} + F(\mathbf{x})$). In LSTMs the skip connection is temporal ($\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$). The additive structure in both cases keeps gradients from vanishing. This connection is not a coincidence — the ResNet authors were aware of the LSTM literature.

```python
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════════
# LSTM FROM SCRATCH — every gate shown explicitly
# This makes the four equations directly visible in code.
# ════════════════════════════════════════════════════════════════════

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # One linear layer computes all four gate inputs at once:
        # [forget, input, candidate, output] concatenated = 4 * hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h_prev, c_prev):
        # Concatenate previous hidden state and current input
        combined = torch.cat([h_prev, x], dim=1)   # (batch, hidden+input)

        # One matrix multiply gives all four pre-activations
        gates = self.linear(combined)              # (batch, 4*hidden)
        f, i, c_tilde, o = gates.chunk(4, dim=1)  # split into four

        # Apply activations
        f       = torch.sigmoid(f)       # forget gate: how much to keep
        i       = torch.sigmoid(i)       # input gate:  how much to write
        c_tilde = torch.tanh(c_tilde)    # candidate:   what to write
        o       = torch.sigmoid(o)       # output gate: what to expose

        # Cell state update: additive → gradient highway through time
        c_new = f * c_prev + i * c_tilde

        # Hidden state: filtered cell state
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

class LSTMSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cell    = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.head    = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim)
        c = torch.zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):             # process one token at a time
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(self.head(h))     # prediction at each step
        return torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)

# ── Test ──────────────────────────────────────────────────────────────
model = LSTMSequence(input_dim=32, hidden_dim=64, output_dim=10)
x = torch.randn(4, 8, 32)   # batch=4, seq_len=8, input=32
out = model(x)
print(f'Input:  {x.shape}')    # (4, 8, 32)
print(f'Output: {out.shape}')  # (4, 8, 10) — prediction at each step

# PyTorch's built-in LSTM (faster, handles batches and multiple layers):
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2,
               batch_first=True, dropout=0.1)
output, (h_n, c_n) = lstm(x)
print(f'nn.LSTM output: {output.shape}')  # (4, 8, 64)
print(f'Final hidden:   {h_n.shape}')     # (2, 4, 64) — 2 layers
print(f'Final cell:     {c_n.shape}')     # (2, 4, 64) — 2 layers
```

*Code 2 – LSTM from scratch. The critical line is `c_new = f * c_prev + i * c_tilde`: the additive structure means gradients flow back through `c` without being multiplied by weight matrices, solving the vanishing gradient problem. PyTorch's `nn.LSTM` uses the same equations but is highly optimised with cuDNN kernels.*

---

## 4  Sequence-to-Sequence Models and the Bottleneck Problem

### 4.1  The Seq2Seq Architecture

Translation, summarisation, and image captioning all require variable-length output from variable-length input — tasks where sequence labelling (one output per input token) is insufficient. The sequence-to-sequence (seq2seq) architecture addresses this with an encoder-decoder structure:

- **Encoder**: An LSTM processes the entire input sequence and compresses it into a single context vector $\mathbf{h}_\text{encoder}$ — the final hidden state after reading all input tokens.
- **Decoder**: A separate LSTM generates the output sequence autoregressively, one token at a time, conditioned on $\mathbf{h}_\text{encoder}$. At each step it predicts the next output token and feeds it back as the next input.

The decoder is a conditional language model: it predicts $P(y_i \mid X, y_1, \ldots, y_{i-1})$, where $X$ is the input and $y_1, \ldots, y_{i-1}$ are the previously generated output tokens. The encoder provides the conditioning through $\mathbf{h}_\text{encoder}$, which seeds the decoder's initial hidden state.

### 4.2  The Information Bottleneck

The fundamental limitation of basic seq2seq is that the entire input — regardless of its length — must be compressed into a single fixed-size vector $\mathbf{h}_\text{encoder}$. For short sequences this is fine. For long sequences (long documents, long sentences) it is catastrophic: as sequence length grows, the encoder must cram more and more information into the same-size vector, and information is inevitably lost. Early parts of the input are particularly likely to be forgotten by the time the encoder finishes processing the final token.

This problem was identified memorably by Ray Mooney: *"You can't cram the meaning of a whole %&!$ing sentence into a single $&!*ing vector!"* The attention mechanism was developed specifically to address this.

---

## 5  The Attention Mechanism

### 5.1  The Core Idea

Rather than forcing the decoder to work from a single context vector, attention allows the decoder to dynamically look back at all encoder hidden states and weight them according to their relevance to the current decoding step. This is the database lookup analogy:

- **Keys**: Representations of each input token (computed from encoder hidden states).
- **Values**: The actual information stored at each input position (also from encoder states, but through a different projection).
- **Query**: What the decoder is currently looking for (computed from the decoder's current hidden state).

The attention computation at each decoder step:

$$a_i = \mathbf{q} \cdot \mathbf{k}_i \qquad \text{(alignment scores — dot product)}$$

$$\alpha_i = \text{softmax}(\mathbf{a})_i \qquad \text{(attention weights — sum to 1)}$$

$$\mathbf{c} = \sum_i \alpha_i \cdot \mathbf{v}_i \qquad \text{(context vector — weighted sum)}$$

The context vector $\mathbf{c}$ is a soft, weighted summary of all input positions, where the weights reflect how relevant each input position is to the current output token. For the translation 'Ich hasse diesen Film' → 'I hate this movie', when decoding 'I', the highest attention weight should be on 'Ich'; when decoding 'hate', the highest weight should be on 'hasse'. The attention mechanism learns these alignments automatically from data.

### 5.2  Self-Attention

In the attention mechanisms described above (cross-attention), queries come from the decoder and keys/values come from the encoder. Self-attention is attention where queries, keys, and values all come from the same sequence. When applied to the encoder's input, each token's embedding is updated to reflect its relationships with all other tokens in the same sentence.

This solves a fundamental problem of plain word embeddings: the same word can mean different things in different contexts. The word 'Apple' in 'I ate an apple' should have a different representation than 'Apple' in 'Apple released a new iPhone'. After self-attention, both encodings of 'Apple' are updated based on context — the surrounding words shift the embedding into the correct meaning subspace.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM — step by step, then as seq2seq
# ════════════════════════════════════════════════════════════════════

def attention(query, key, value, mask=None):
    """
    Scaled dot-product attention.
    query: (batch, n_q, d_k)
    key:   (batch, n_k, d_k)
    value: (batch, n_k, d_v)
    Returns: context (batch, n_q, d_v), weights (batch, n_q, n_k)
    """
    d_k = query.size(-1)

    # Step 1: alignment scores (dot product of query with every key)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_k)

    # Step 2: optional masking (set future positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 3: normalise to attention weights
    weights = F.softmax(scores, dim=-1)   # (batch, n_q, n_k) rows sum to 1

    # Step 4: weighted sum of values
    context = torch.bmm(weights, value)   # (batch, n_q, d_v)

    return context, weights

# ── Concrete example: translating 'Ich hasse' → 'I hate' ─────────────
batch, d = 1, 4

# Simulated encoder hidden states for ['Ich', 'hasse']
encoder_states = torch.randn(batch, 2, d)   # (1, 2, 4)

# Keys and Values come from encoder (through linear projections)
W_k = nn.Linear(d, d, bias=False)
W_v = nn.Linear(d, d, bias=False)
W_q = nn.Linear(d, d, bias=False)

keys   = W_k(encoder_states)   # (1, 2, 4)
values = W_v(encoder_states)   # (1, 2, 4)

# Query comes from decoder (current step's hidden state)
decoder_state = torch.randn(batch, 1, d)  # (1, 1, 4)
query = W_q(decoder_state)                # (1, 1, 4)

context, attn_weights = attention(query, keys, values)
print(f'Attention weights: {attn_weights[0, 0].detach().tolist()}')
# e.g. [0.73, 0.27] — 'Ich' gets most attention when decoding 'I'
print(f'Context vector:   {context.shape}')   # (1, 1, 4) — weighted sum

# ── Seq2Seq with attention ────────────────────────────────────────────
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed       = nn.Embedding(vocab_size, embed_dim)
        self.encoder     = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder     = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                                   batch_first=True)
        # Projections for key, value, query
        self.W_k   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src):
        emb  = self.embed(src)                   # (B, src_len, embed_dim)
        enc_out, (h, c) = self.encoder(emb)      # enc_out: (B, src_len, H)
        return enc_out, h, c

    def decode_step(self, tgt_token, h, c, enc_out):
        # Embed current target token
        emb = self.embed(tgt_token)              # (B, 1, embed_dim)

        # Compute attention over encoder outputs
        query   = self.W_q(h.transpose(0,1))     # (B, 1, H)
        keys    = self.W_k(enc_out)              # (B, src_len, H)
        values  = self.W_v(enc_out)              # (B, src_len, H)
        ctx, w  = attention(query, keys, values) # (B, 1, H)

        # Concatenate embedding with context vector, run decoder LSTM
        dec_input       = torch.cat([emb, ctx], dim=-1)
        dec_out, (h, c) = self.decoder(dec_input, (h, c))
        logits          = self.output(dec_out)   # (B, 1, vocab)
        return logits, h, c, w

    def forward(self, src, tgt):
        enc_out, h, c = self.encode(src)
        all_logits, all_weights = [], []
        for t in range(tgt.size(1)):
            logits, h, c, w = self.decode_step(
                tgt[:, t:t+1], h, c, enc_out)
            all_logits.append(logits)
            all_weights.append(w)
        logits  = torch.cat(all_logits, dim=1)   # (B, tgt_len, vocab)
        weights = torch.cat(all_weights, dim=1)  # (B, tgt_len, src_len)
        return logits, weights

model = Seq2SeqWithAttention(vocab_size=1000, embed_dim=64, hidden_dim=128)
src = torch.randint(0, 1000, (2, 5))   # 2 sentences, 5 src tokens
tgt = torch.randint(0, 1000, (2, 4))   # 4 target tokens
logits, weights = model(src, tgt)
print(f'Output logits:     {logits.shape}')   # (2, 4, 1000)
print(f'Attention weights: {weights.shape}')  # (2, 4, 5) src-tgt alignment
```

*Code 3 – Attention from scratch, then wired into a complete seq2seq model. The attention weights matrix (`tgt_len × src_len`) is the alignment matrix — visualising it shows which source tokens the model attended to when generating each target token, providing a direct window into what the model has learned.*

---

## 6  From Seq2Seq to Transformers

### 6.1  The Sequential Processing Bottleneck

RNNs and LSTMs with attention are expressive and work well, but they have a fundamental architectural constraint: processing is sequential. To compute $\mathbf{h}_5$, you must first compute $\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3, \mathbf{h}_4$. This serial dependency means RNNs cannot be parallelised across the time dimension, which is a severe bottleneck for long sequences on modern GPU hardware that excels at massively parallel computation.

### 6.2  Removing the RNN

The key insight behind the Transformer (Vaswani et al., 2017): if we have self-attention, we do not need the RNN at all. Self-attention already allows every token to communicate with every other token in a single operation — it is the RNN's role as an information aggregator that attention fulfils, but without the sequential dependency. By removing the LSTM and keeping only the attention operations, all tokens in a sequence can be processed in parallel.

The Transformer encoder replaces the LSTM sequence with:

- **Multi-head self-attention** — multiple attention heads learn different relationship patterns.
- **Position-wise feed-forward networks** — add non-linearity per token.
- **Residual connections and layer normalisation** — enable deep stacking.
- **Positional encodings** — inject sequence order information (since self-attention is permutation-invariant).

The Transformer decoder adds masked self-attention (to prevent attending to future tokens during autoregressive generation) and cross-attention from decoder queries to encoder keys/values. This architecture is covered in detail in Lecture 12.

| Property | Vanilla RNN | LSTM | Transformer |
|---|---|---|---|
| Long-range deps. | Poor (vanishing grad) | Good (cell state highway) | Excellent (direct attention) |
| Parallelism | Sequential only | Sequential only | Fully parallel |
| Memory (in $O$) | $O(1)$ per step | $O(1)$ per step | $O(n^2)$ attention matrix |
| Context window | Theoretically $\infty$ (but practical ~10–20) | Long (~100s) | Fixed (e.g. 2048 tokens) |
| Training speed | Slow | Moderate | Fast on GPU/TPU |

---

## 7  Image Captioning: Bringing Vision and Language Together

Image captioning is the application that unifies everything in this lecture. It is a one-to-many sequence problem: the input is a single image; the output is a sequence of words describing it. It requires both visual understanding (the CNN or ViT encoder) and language generation (the LSTM or Transformer decoder).

### 7.1  CNN-LSTM Captioning

The classical approach uses a CNN backbone (AlexNet, VGG, ResNet) to produce a fixed-dimensional image embedding $\mathbf{h}_\text{CNN}$, which seeds the decoder LSTM's initial hidden state. The decoder then generates words autoregressively, one at a time, conditioned on $\mathbf{h}_\text{CNN}$:

$$P(\text{caption} \mid \text{image}) = \prod_i P(y_i \mid \mathbf{h}_\text{CNN},\, y_1, \ldots, y_{i-1})$$

This is exactly the conditional language model from Section 4.1, with $\mathbf{h}_\text{CNN}$ playing the role of $\mathbf{h}_\text{encoder}$. Training uses teacher forcing: at each decoder step, the ground-truth previous word is fed in rather than the model's own prediction, which stabilises training. At inference time, the model's own previous output is used (autoregressive decoding).

### 7.2  Captioning with Spatial Attention

The limitation of basic CNN-LSTM captioning: the image is compressed to a single vector before any decoding. As with text seq2seq, this loses spatial information. A richer approach extracts spatial feature maps from the CNN (shape $H \times W \times D$, one feature vector per spatial location), then applies attention over these spatial features at each decoding step.

When generating the word 'hat' in 'man wearing straw hat', the attention weights should highlight the hat region of the image. When generating 'straw', they should again focus on that region but perhaps with a different pattern. This spatial attention provides both better captions and interpretable visualisations of what the model is attending to at each word — the model is explicitly showing its work.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── CNN-LSTM Image Captioning with Spatial Attention ──────────────────
class ImageCaptioner(nn.Module):
    """
    Image captioning with spatial attention.
    Encoder: CNN extracts spatial feature map (H×W×D).
    Decoder: LSTM attends to spatial features at each word step.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 feat_dim=2048):
        super().__init__()

        # ── Visual encoder ────────────────────────────────────────────
        # ResNet50 backbone; remove final FC and pooling to keep spatial map
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # → (B,2048,7,7)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)   # project to hidden_dim

        # ── Attention ─────────────────────────────────────────────────
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)  # query from decoder
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  # key from visual
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)  # value from visual

        # ── Language decoder ──────────────────────────────────────────
        self.embed   = nn.Embedding(vocab_size, embed_dim)
        # LSTM input: word embedding + attended visual context
        self.decoder = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, vocab_size)

    def encode_image(self, image):
        feat = self.cnn(image)                       # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1)              # (B, 7, 7, 2048)
        feat = feat.view(B, H*W, C)                  # (B, 49, 2048) — 49 spatial locs
        feat = self.feat_proj(feat)                  # (B, 49, hidden_dim)
        return feat

    def attend(self, h, visual_feat):
        query = self.W_q(h.unsqueeze(1))             # (B, 1, H)
        keys  = self.W_k(visual_feat)                # (B, 49, H)
        vals  = self.W_v(visual_feat)                # (B, 49, H)
        scores  = (query * keys).sum(-1)             # (B, 49) — dot product
        weights = torch.softmax(scores, dim=-1)      # (B, 49) — sum to 1
        context = (weights.unsqueeze(-1) * vals).sum(1)  # (B, H)
        return context, weights

    def forward(self, images, captions):
        visual_feat = self.encode_image(images)      # (B, 49, H)
        # Initialise hidden state as mean of spatial features
        h = visual_feat.mean(1)                      # (B, H)
        c = torch.zeros_like(h)

        logits_all, attn_all = [], []
        for t in range(captions.size(1)):
            word_emb          = self.embed(captions[:, t])   # (B, embed_dim)
            context, attn_w   = self.attend(h, visual_feat)  # (B, H), (B, 49)
            decoder_input     = torch.cat([word_emb, context], dim=1)
            h, c              = self.decoder(decoder_input, (h, c))
            logits_all.append(self.output(h).unsqueeze(1))
            attn_all.append(attn_w.unsqueeze(1))

        logits = torch.cat(logits_all, dim=1)    # (B, seq_len, vocab)
        attn   = torch.cat(attn_all,   dim=1)    # (B, seq_len, 49)
        # attn[:, t, :] reshaped to (7, 7) shows which image region
        # the model attended to when generating word t
        return logits, attn

captioner = ImageCaptioner(vocab_size=5000, embed_dim=256, hidden_dim=512)
images   = torch.randn(2, 3, 224, 224)        # 2 images
captions = torch.randint(0, 5000, (2, 10))    # 10 word captions
logits, attn = captioner(images, captions)
print(f'Output logits:     {logits.shape}')   # (2, 10, 5000)
print(f'Attention maps:    {attn.shape}')     # (2, 10, 49)
print(f'Reshape attn[t]:   {attn[0,0].view(7,7).shape} — spatial heatmap per word')
```

*Code 4 – CNN-LSTM image captioning with spatial attention. The attention map `attn[:, t, :]` can be reshaped to $(7, 7)$ and overlaid on the input image to visualise which spatial region the model was looking at when it generated word $t$. This produces the famous attention visualisations from Xu et al. (2015) 'Show, Attend and Tell'.*

---

## 8  Language Model Families: BERT, GPT, and BART

Modern large language models pre-train on vast text corpora and fine-tune on specific tasks. Three architectural families, each with a different pre-training objective:

| Model | Architecture | Pre-training task | Strengths | Example tasks |
|---|---|---|---|---|
| BERT | Encoder only | Masked token prediction (bidirectional) | Rich bidirectional representations | Classification, NER, QA |
| GPT | Decoder only | Next token prediction (autoregressive) | Text generation, in-context learning | Completion, chatbots, code |
| BART/T5 | Encoder-decoder | Denoising (mask, shuffle, drop) | Versatile: understands and generates | Translation, summarisation |

GPT-3 (Brown et al., 2020) demonstrated that sufficiently large models trained purely on next-token prediction develop in-context learning: they can perform new tasks by reading a few examples in the prompt, without any gradient updates. This works because the diversity of pre-training data means the model has effectively learned to learn from examples. In-context learning (also called few-shot prompting) revolutionised how language models are deployed — instead of fine-tuning for every task, you describe the task in natural language.

---

## 9  Summary: The Evolutionary Arc

This lecture traced a single line of reasoning from vanilla RNNs to the Transformer, with each step motivated by the failure of the previous one:

| Architecture | Key mechanism | Motivated by |
|---|---|---|
| Vanilla RNN | Shared hidden state $\mathbf{h}_t$ | MLPs cannot handle variable-length sequences or context |
| LSTM | Cell state + gating mechanism | Vanilla RNN: vanishing gradient prevents long-range dependencies |
| Seq2Seq | Encoder-decoder with fixed context | LSTM: no architecture for variable-length output |
| Seq2Seq + Attn | Soft alignment over all encoder states | Seq2Seq: single context vector is an information bottleneck |
| Transformer | Self-attention replaces recurrence | LSTM + attention: sequential processing limits parallelism |

Two themes connect this entire lecture to the broader course. First, the inductive bias trade-off: RNNs bake sequential processing into their architecture; Transformers make no such assumption and let attention patterns emerge from data. As with CNNs vs ViTs, the less-biased model wins given enough data. Second, the encoder-decoder pattern: every architecture here (seq2seq, image captioning, BERT fine-tuning, Stable Diffusion's U-Net with cross-attention) follows the same structure — a powerful encoder compresses the input into rich representations; a task-specific decoder reads those representations to produce the desired output. Mastering this pattern is the key to understanding and building modern deep learning systems.

---

## References

- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Bahdanau, D. et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.
- Xu, K. et al. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners (GPT-3). NeurIPS.
- Illustrated Transformer: jalammar.github.io/illustrated-transformer/
- CMU Advanced NLP course: phontron.com/class/anlp2022/