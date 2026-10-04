# Iteration Log — Week 7 pages
### Deep Learning for Visual Recognition — Optimisation and Regularisation

**Group number:**

---

## Instructions

Add these pages to the **running iteration log** you started in Week 6. Work through the lab
notebook (`Lab7_OptimisationRegularisation.ipynb`) first, then complete Parts A–D for **your own
project** with your group. Discuss Part D with a TA before you leave.

**The main rule for this week:** the lecture gives you a toolbox of more than ten techniques. Do not
switch them all on. Every change you make to your project must be justified by something you
**observed** (Part A) and tested as a **single change** (Part D).

**Don't transfer the lab's results to your project.** The notebook used a toy network, a toy dataset
and short training budgets. A technique that didn't improve accuracy in the lab may well help in your
project, and vice versa. What you should transfer is the method from Part 5 of the notebook: for each
change, predict and measure both **(a) the mechanism** (does the technique do what it is designed to
do?) and **(b) the outcome** (does your primary metric improve?), and explain any difference between
the two.

**A note on task type:** most techniques apply to every task. Some need adapting: for example,
Mixup and label smoothing are defined for classification. For segmentation, augmentation must
transform the image and the mask *together*; for detection, the image and the boxes. If something
genuinely doesn't apply, write **N/A** and a one-sentence reason.

---

# Part A — Diagnose Your Own Training Curves

Train your current best model (from your Week 6 iteration log) with **early stopping disabled**
for clearly longer than usual, and log the training and validation loss **and** your primary
metric every epoch. If a full run is too expensive, use the cheap screening setup from your Week 6
budget (subset, lower resolution) and say so.

| Quantity | Value |
|---|---|
| Epoch with the lowest validation loss | |
| Epoch with the best validation metric | |
| Training loss at the end (measured **without** augmentation/dropout, model in eval mode) | |
| Validation loss at the end | |
| Primary metric: best / final | |

**Which pattern from the lecture (slides 87–93) do your curves show?**

- [ ] Overfitting: training loss keeps falling, validation loss rises
- [ ] Underfitting: both losses high and flat
- [ ] Optimisation problem: loss not decreasing, very noisy, or diverging
- [ ] Training still improving when it stopped (too short)
- [ ] Healthy: small gap, both curves have converged
- [ ] Other: ______________________________

**Evidence** (attach or describe the plot):

**Does the best-loss epoch differ from the best-metric epoch? What could that mean for your task?**
(Remember Part 1 of the notebook: overconfidence.)

---

# Part B — From Diagnosis to Candidate Remedies

Use your Part A diagnosis to fill in the table. **Only list remedies that address what you
observed.** For each one, say why it should help *your* project, and what could make it fail.

| Candidate remedy | Which observation does it address? | Why it might help here | Why it might not (risk, cost, task mismatch) |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**One technique from the lecture that you deliberately will *not* use, and why:**

*Augmentation check (if augmentation is a candidate):* which transformations preserve your labels?
(E.g., horizontal flips are wrong for digits and text, rotations may be wrong for some medical or
satellite images, colour jitter may be wrong when colour is the signal.) List the ones you will use
and the ones you rule out.

| Augmentation | Label-preserving for our task? | Reason |
|---|---|---|
| | Yes / No | |
| | Yes / No | |
| | Yes / No | |

---

# Part C — Optimiser, Learning Rate and Hyperparameter Search Plan

## 1. Current training setup

| Item | Current choice | Justified by (evidence or reference)? |
|---|---|---|
| Optimiser | | |
| Learning rate | | |
| Schedule / warm-up | | |
| Batch size | | |
| Early stopping (on which metric, patience) | | |

**Have you run an LR range test on your model?** If not, run one (Part 3 of the notebook): it costs
only a few hundred steps. Result:

## 2. Hyperparameter search plan

List the hyperparameters you intend to tune, **most important first** (usually the learning rate).

| Hyperparameter | Range | Scale (linear / log) | Number of values (coarse stage) |
|---|---|---|---|
| | | | |
| | | | |

**Search strategy:** grid / random / coarse-to-fine (slides 76–83). Why?

**Budget check** (from your Week 6 budget): number of runs this search needs × time per run =

If this exceeds your budget, what will you cut? (Fewer values, fewer epochs in the coarse stage,
a data subset, or fewer hyperparameters.)

**How will you detect a best value at the edge of a range?**

---

# Part D — Iteration I2: The First Experiment After the Autumn Break

Copy the iteration page template from your Week 6 log and fill in steps 1–4 now. Step 5
(evaluation) is done after the break.

## Iteration I2  ·  Date: __________

### 1. Clarification — issues (from Part A)

| # | Issue | Evidence |
|---|---|---|
| 1 | | |
| 2 | | |

### 2. Prioritization — the issue we tackle first, and why

### 3. Analysis — possible causes, and how we checked them

| Possible cause | How we checked it | What we found |
|---|---|---|
| | | |

### 4. Design solutions — hypothesis and planned experiment

**Hypothesis:** "We expect ___ because ___."

**The single change** (from Part B), **compared with which experiment:**

**What is kept fixed:**

**Seeds / runs planned, and the time they will take:**

**✍️ Prediction:**

| Metric | Reference experiment | Predicted value |
|---|---|---|
| | | |

**How we will compare:** best epoch (early stopping) or a fixed number of epochs? Why?

### 5. Evaluation — *after the break*

(Leave empty for now.)

---

# Part E — Status Before the Break

| Item | Status |
|---|---|
| Pipeline passes the four sanity checks (Week 6) | ☐ |
| Baseline (E0) trained, with ≥ 2–3 seeds or a justified alternative | ☐ |
| Experiment log in use for every run | ☐ |
| Iteration I1 evaluated (Week 6) | ☐ |
| Training curves diagnosed (Part A) | ☐ |
| I2 planned (Part D) | ☐ |
| Test set untouched so far | ☐ |

**The first thing we will do after the break:**

**Anything blocking us that the TAs should know about** (data access, compute, unclear scope)?

---