# Iteration Log
### Deep Learning for Visual Recognition — Week 6 onwards

**Group number:**

---

## Instructions
This document is meant as a tool to help you log your experiments. It is **not mandatory** that you use it, but it is recommended that you work through it at today' lab.

You can think of it as a **running document**. Today you complete Parts A and B and
your **first iteration page** (Part C). For the rest of the course, add a new iteration page
every time you go around the loop — normally at least once a week. Keep it in your group's
shared folder or repository and bring it to every lab session.

Work through the lab notebook (`Lab6_IterativeWorkflow.ipynb`) first, then do Parts A–C with
your group. Discuss your first iteration page with a TA before you leave.

**Why keep a log?** Your final report must explain not only *what* you did but *why* you did
it. A log written at the time is far more reliable than your memory in December. Each iteration
page maps directly onto your report:

| Iteration page | Report section |
|---|---|
| Issue, analysis and hypothesis | Methods (why each design decision was made) |
| Result table | Results |
| Conclusion, including negative results | Discussion |

**A note on task type:** the notebook uses classification, but the loop is the same for every
task. Where a question needs adapting, notes are given. If something genuinely doesn't apply,
write **N/A** and a one-sentence reason rather than leaving it blank.

---

# Part A — Can You Trust Your Pipeline?

Consider the four sanity checks from Part 1 of the notebook. You should have similar sanity checks on **your own** data and model. Re-run them whenever you make a major change to your code (new data loading, new model, new loss).

Today, try to run the sanity checks from the notebook on your data, and fill in the table below.

| Check | What we expected | What we observed | Pass? |
|---|---|---|:---:|
| **A. Look at the data:** images and labels look right, normalised stats ≈ 0/1 in all splits | | | ☐ |
| **B. Check the split:** no overlap and no grouping leakage (same patient, video, object…; see your Week 2 and 3 worksheets) | | | ☐ |
| **C. Initial loss** matches the expected value | | | ☐ |
| **D. Overfit one batch:** loss → ≈ 0, predictions match the targets | | | ☐ |

*Adapting check C:* classification or per-pixel segmentation with cross-entropy → ln(C).
Regression → the loss of always predicting the mean target. Detection → there is no simple
expected value; check instead that the individual loss terms (classification, box regression)
start at sensible magnitudes and are not NaN.

*Adapting check D:* for segmentation, the predicted masks on the batch should match the
ground truth almost perfectly. For detection, the boxes should. For generation, the model
should be able to reproduce (or reconstruct) a handful of training images.

**Bugs we found and fixed (if any):**

---

# Part B — Baseline, Noise Floor and Budget

## 1. Current baseline (E0)

| Item | Value |
|---|---|
| Model / pretrained weights | |
| What is trained (all layers, head only, …) | |
| Key hyperparameters (learning rate, batch size, epochs, optimiser) | |
| How the learning rate was chosen (sweep? default?) | |
| Primary metric (from your Week 4 worksheet) | |

## 2. Noise floor

Run the baseline with at least three different seeds, or plan when you will.

| Seed | Primary metric (val.) | Secondary metric (val.) |
|---|---|---|
| 0 | | |
| 1 | | |
| 2 | | |
| **Mean ± std** | | |

**The smallest difference we will treat as meaningful** (roughly 2 × std):

*If three seeds per configuration are too expensive:* explain how you will deal with this — for
example, three seeds only for the final comparisons, or treating small differences as
inconclusive.

## 3. Experiment budget

| Question | Answer |
|---|---|
| Time per epoch on our full data (measured, not guessed) | |
| Epochs per run | |
| Time per run | |
| Realistic GPU hours available until the deadline | |
| **Runs we can afford** (÷ number of seeds per configuration) | |
| How we will make runs cheaper for screening experiments (subset, resolution, fewer epochs, …) | |
| Where checkpoints are saved | |

## 4. Experiment log

Where does your group log every run (CSV file, spreadsheet, W&B, …)? What is recorded for each
run? At minimum: experiment name, full configuration, seed, metrics, runtime and date.

---

# Part C — Iteration Pages

Copy the template below for each new iteration. Number them I1, I2, I3, … Keep the old ones —
**including the ones where the hypothesis turned out to be wrong**. Those are often the most
useful material for your Discussion section.

The five steps follow the workflow from the project introduction:
Clarification → Prioritization → Analysis → Design solutions → Evaluation.

---

## Iteration I__  ·  Date: __________

### 1. Clarification — what issues do we see with our current setup?

List everything you have noticed, with the evidence. "Val. accuracy is low" is not an issue
description. "Training loss keeps falling but validation loss rises after epoch 4" is.

| # | Issue | Evidence (curve, metric, figure, observation) |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

### 2. Prioritization — which issue do we tackle now, and why?

**Selected issue:**

**Why this one first?** (For example: it blocks other experiments, it has the largest effect on
the research question, or it is cheap to test.)

### 3. Analysis — what could be causing it?

List possible causes, and what you did to tell them apart. Useful tools: loss curves, confusion
matrix or per-class metrics, the most confidently wrong predictions, activation histograms,
gradient flow, Grad-CAM, or re-evaluating on a subset (for example, only the small objects).

| Possible cause | How we checked it | What we found |
|---|---|---|
| | | |
| | | |

### 4. Design solutions — hypothesis and planned experiment

**Hypothesis:** "We expect ___ because ___."

**The single change we will make** (compared with which previous experiment?):

**What is kept fixed:**

**✍️ Prediction, written before running:**

| Metric | Reference experiment | Predicted value | Actual value |
|---|---|---|---|
| | | | |
| | | | |

**Seeds / runs planned:**

### 5. Evaluation — what happened?

| Experiment | Change | Primary metric (mean ± std, n seeds) | Secondary metric | Notes |
|---|---|---|---|---|
| E0 (baseline) | — | | | |
| | | | | |

**Is the difference larger than our noise floor?** Yes / No / Inconclusive

**Conclusion:** is the hypothesis supported, rejected, or is the evidence inconclusive?

**One sentence for the Results section** (objective, no interpretation):

**One sentence for the Discussion section** (interpretation — why did it work or not?):

**Next step:** keep this change? What new issues appeared? → take them to the next iteration.

---

# Part D — Running Summary Tables

Update these as you go. Together they become the core of your Results section.

## Experiment ladder (adding one change at a time)

| Exp. | Change from previous / from baseline | Question it answers | Primary metric (mean ± std) | Iteration page |
|---|---|---|---|---|
| E0 | Baseline | How well does the basic approach work? | | |
| E1 | | | | |
| E2 | | | | |
| E3 | | | | |

## Ablation of the final model (removing one component at a time)

Fill this in once you have settled on a final model — usually in the last weeks of the project.

| Model | Primary metric (mean ± std) | Change vs. full model |
|---|---|---|
| Full model | | — |
| − component 1: | | |
| − component 2: | | |
| − component 3: | | |

**Did you find any interactions** — components whose effect depends on whether another component
is present (like Kaiming initialisation and batch normalisation in the notebook)?

---

# Generic check list

| Item | Needs work | Plausible | Clear |
|---|:---:|:---:|:---:|
| Pipeline sanity checks done and passing | ☐ | ☐ | ☐ |
| Baseline runs and noise floor is known or planned | ☐ | ☐ | ☐ |
| Budget is realistic for the planned experiments | ☐ | ☐ | ☐ |
| Issue is backed by evidence (not a guess) | ☐ | ☐ | ☐ |
| Hypothesis is testable with a single, controlled change | ☐ | ☐ | ☐ |
| Prediction was written before the experiment | ☐ | ☐ | ☐ |
