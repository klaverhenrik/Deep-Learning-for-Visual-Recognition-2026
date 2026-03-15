# Results and Figures
### Deep Learning for Visual Recognition — Week 11

This is the last lab session before the final push. The session is an open
supervised work session — bring your current results and your draft report,
and use the time to get feedback from TAs on whatever you need most.

The first 15 minutes of the session are structured around one topic:
**how to present your results clearly**.

---

## The goal of a results section

Your results section has one job: convince the reader that your approach
works, and that you understand *why* it works. Every figure, table, and
number should serve that goal. If it does not, cut it.

---

## Figures

### What makes a good figure

A figure should be self-contained — a reader should be able to understand
it without reading the surrounding text.

**Required elements for every figure:**
- Clear axis labels with units
- A title or caption that describes what is being shown
- A legend if more than one series is plotted
- Appropriate scale (log scale for loss curves, linear for accuracy)

**Common figure mistakes:**
- Axis labels missing or too small to read
- Legend entries that say "Series 1" instead of something meaningful
- Lines so close in colour or style they cannot be distinguished in print
- Figures that are too small — if it does not fill at least a third of
  the page width, it is too small

### Which figures belong in your report

| Figure | Required? | What it shows |
|---|---|---|
| Loss curves (train + val) | Yes | Evidence that training converged |
| Confusion matrix | Yes (classification) | Per-class performance pattern |
| Results table | Yes | Summary of all experiments |
| Worst-case examples | Recommended | Where the model fails and why |
| Ablation bar chart | Yes (if you ran ablations) | Contribution of each component |
| Grad-CAM / t-SNE | Optional | Model interpretability |
| Example outputs | Yes (generation/detection/segmentation) | Qualitative evidence |

### Loss curves checklist

Before including a loss curve in your report:
- [ ] Both training and validation curves are shown
- [ ] X axis is labelled "Epoch" or "Step"
- [ ] Y axis is labelled "Loss" (or the specific loss function name)
- [ ] The curve shows convergence — it should not still be steeply falling
  at the last epoch shown
- [ ] If comparing multiple runs, they are on the same axes

---

## Tables

### Results table format

Every number in your results table should be:
- Measured on the **validation set** during development
- Measured on the **test set** for your final reported result (once)
- Reported to **3 significant figures** (e.g. 0.923, not 0.9234567)
- Accompanied by the metric name (accuracy, macro F1, mAP...)

### What belongs in the table

| Column | Required? | Notes |
|---|---|---|
| Model / condition name | Yes | Be specific — "ResNet-18 fine-tuned" not just "ours" |
| Primary metric | Yes | Consistent across all rows |
| Secondary metric | Recommended | Especially for imbalanced classes |
| Number of parameters | Optional but useful | Shows model complexity |
| Training time | Optional | If relevant to your claim |

### Formatting rules

- Bold the best result in each column
- Use horizontal lines to separate trivial baselines from trained models
- If comparing to prior work, note if they used a different dataset split
- Include random chance and your simplest baseline as the first rows

---

## Common report writing mistakes to avoid

**Reporting training accuracy instead of validation accuracy.**
Training accuracy is almost always higher and tells you nothing
about generalisation.

**Reporting the best validation result across many runs.**
If you ran 50 experiments and report the best one, you are
reporting a lucky outlier. Report the result from your final,
deliberately chosen model.

**Figures without captions.**
Every figure needs a caption below it. The caption should describe
what is shown and what the reader should take away from it.

**"The model achieved X% accuracy" without context.**
X% accuracy on what? Compared to what? With what metric?
Always provide the context.

**Forgetting to describe failure cases.**
A results section that only shows successes is not credible.
Show your failure cases and explain them — this demonstrates
understanding.

---

## Today's session

**First 15 minutes:** read through this guide and apply it to your
current figures and tables. Fix anything that fails the checklists above.

**Remaining time:** open work session.

Suggested priorities for today:
1. Generate all the figures you need for your final report
2. Draft your results section using the table template
3. Get TA feedback on your current results and what is missing

**A TA will come to your group at least once during the session.**
Come prepared with a specific question or something you want feedback on.
"Does this look right?" is a good question. "Can you read our draft?"
is also a good question. "We don't know what to work on" is a signal
to ask for help sooner rather than later.
