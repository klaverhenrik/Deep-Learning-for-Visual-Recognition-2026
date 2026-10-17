# Iteration Log — Week 8 pages
### Deep Learning for Visual Recognition — Operating on a Pretrained Network

**Group number:**

---

## Instructions

Add these pages to your **running iteration log** (started in Week 6). Work through the lab notebook
(`Lab8_PretrainedNetworkSurgery.ipynb`) first. Then complete Parts A–C with your group, and discuss
Part B with a TA before you leave.

**The idea of this week:** the networks from the lecture are not fixed recipes. You can change their
inputs, their outputs, what is trained and how. Each such change can become a **controlled experiment**
that answers a question about your problem. This week's pages are about finding those questions for
your own project.

**As in Lab 7:** don't transfer the lab's *numbers* to your project. The lab used a small SVHN subset
and short training runs, so "LoRA beat the last block" or "grayscale was as good as RGB" are findings
about that setup, not rules. Transfer the *operations* and the *way of reasoning*.

---

# Part A — Where Are You Now?

## 1. Iteration I2 (planned in Week 7)

If you haven't done so already, fill in step 5 (Evaluation) of iteration I2 in your Week 7 pages.

| | Predicted | Observed (mean ± std, paired difference) | Consistent across seeds? |
|---|---|---|---|
| Mechanism | | | |
| Outcome | | | |

**Conclusion, and what it means for the next iteration:**

## 2. Your current model

| Item | Answer |
|---|---|
| Architecture / pretrained weights | |
| What is trained (everything, the head only, some stages…) | |
| Input (channels, resolution, preprocessing / normalisation) | |
| Output (task, head) | |
| Best result so far (primary metric, mean ± std) | |

**Does your preprocessing match what the pretrained network expects** (normalisation statistics,
number of channels, roughly the right resolution)? If not, why not, and is that a deliberate choice?

---

# Part B — Operations on Your Network

Pick **two** operations from the idea generator in Part 4 of the notebook (or invent your own) that
could make sense for your project. The goal is not "something that might raise accuracy", but **a
question that the operation answers**.

*Examples of questions:* Does our model need colour (or this band, or this metadata field)? Is the
model using a shortcut (a ruler, a watermark, the scanner type)? Where in the network does the
adaptation to our domain happen? Can we get the same result training 2% of the parameters? Does a
second task help the first?

## Idea 1

| Item | Answer |
|---|---|
| Operation (input / output / fusion / what to train / trained weights) | |
| **The question it answers** | |
| The single change, compared with which baseline | |
| Mechanism you can measure, and ✍️ prediction | |
| Outcome (primary metric), and ✍️ prediction | |
| What would you conclude if the outcome doesn't change? | |
| Cost: implementation time; number of runs × time per run (from your Week 6 budget) | |

## Idea 2

| Item | Answer |
|---|---|
| Operation (input / output / fusion / what to train / trained weights) | |
| **The question it answers** | |
| The single change, compared with which baseline | |
| Mechanism you can measure, and ✍️ prediction | |
| Outcome (primary metric), and ✍️ prediction | |
| What would you conclude if the outcome doesn't change? | |
| Cost: implementation time; number of runs × time per run (from your Week 6 budget) | |

## Should you consider one of these ideas for your next iteration, and why?

Consider: which question matters most for your research question, which one you can afford,
and which one would still give you something to write about if the result is negative.

**Chosen idea:**

**Reason:**

---

**Keep these pages in your iteration log. Ideas you don't pursue now are good material for the
"Future work" paragraph of your Discussion section.**
