# Model Evaluation Worksheet
### Deep Learning for Visual Recognition — Week 4

**Group number:**

**Project title (from week 2/3):**

**Date:**

---

## Instructions

Today's session has three parts: a short lecture on how to write a good report, the three
evaluation notebooks (`Lab4a_ImageClassification.ipynb`, `Lab4b_ImageSegmentation.ipynb`,
`Lab4c_ObjectDetection.ipynb`), and this worksheet.

Complete the notebook(s) first, then Parts 1–3, discuss with a TA, then finish Parts 4–6.

**A note on task type:** the three notebooks cover classification, segmentation, and detection.
If your project is a different task (generation, regression, retrieval/verification, or
vision-language), use the adaptation notes under each question to figure out the equivalent for your own task. If a question genuinely doesn't apply, write **N/A** and a one-sentence reason
rather than leaving it blank.

## Purpose of this exercise

By the end of the lab, your group should have:

- a primary and secondary evaluation metric chosen for your project, with a stated reason;
- a plan for the `torchmetrics` classes (or equivalent) you'll use to compute them;
- an awareness of at least one way your metric could be computed incorrectly or misleadingly, and
  how you'll guard against it;
- a first draft, in plain language, of how your results will be described in your report's
  Results and Discussion sections.

This worksheet is not asking for final numbers — you likely don't have a trained model yet. It's
asking you to decide *how* you will measure success before you start running experiments, so that
your week 2 research question and your week 3 dataset actually connect to something measurable.

---

# Part 1 — Match Your Project to a Task Type and Notebook

## 1. Which primary task type does your project use?

(Same list as your week 2 worksheet — copy your answer from there if it hasn't changed.)

- [ ] Image classification
- [ ] Object detection
- [ ] Semantic or instance segmentation
- [ ] Image generation
- [ ] Regression
- [ ] Retrieval, matching, or verification
- [ ] Vision-language or multimodal understanding
- [ ] Other: ______________________________

## 2. Which of today's notebooks is the closest match?

- [ ] Lab 4a (classification) — top-1/top-5 error, confusion matrix, precision-recall, AP
- [ ] Lab 4b (segmentation) — IoU, Dice
- [ ] Lab 4c (detection) — IoU, mAP
- [ ] None exactly — see adaptation note below

**If none exactly, what's the closest analogy, and why?**

> Examples: generation has no single ground-truth answer per input, so evaluation usually mixes a
> quantitative proxy (e.g. FID, or a downstream task accuracy) with qualitative inspection — see
> the "How about GANs?" slide from today's lecture. Retrieval/verification tasks often reuse
> precision-recall curves and AP almost exactly as in Lab 4a, just computed over ranked matches
> instead of class scores. Regression has no discrete classes at all — report an error metric
> (MAE/RMSE) and a scatter or residual plot instead of a confusion matrix. Vision-language tasks
> (captioning, VQA) typically need a metric neither notebook covers (e.g. BLEU/CIDEr, or
> task accuracy) — flag this as something to research further this week if it applies to you.

---

# Part 2 — Choosing Your Metrics

## 3. Primary and secondary metrics

| Evaluation component | Your choice | Why is it appropriate for your task and research question? |
|---|---|---|
| **Primary metric** | | |
| **Secondary metric(s)** | | |
| Per-class, subgroup, or robustness breakdown | | |
| Qualitative evaluation plan (see Part 4) | | |

**Sanity-check baseline:** what would this metric look like for a trivial baseline (predicting the
majority class, guessing randomly, or predicting the mean)? You need this number to know whether
your model is actually learning anything.

> Our sanity-check baseline and its expected metric value:

## 4. If your primary metric has a "which threshold/level" choice, which will you report — and why?

This applies directly if you're doing detection or segmentation; for other tasks, look for the
equivalent (e.g. classification threshold for a binary problem, top-k for retrieval).

- **Detection:** will you report `map_50`, `map_75`, the full COCO-style `map` (averaged over
  IoU 0.5–0.95), or several of these? Which one best matches what "success" means for your
  project — is a loosely-correct box acceptable, or does your application need precise
  localization?
- **Segmentation:** IoU, Dice, or both? Per-class or averaged over classes?
- **Classification:** top-1, top-5, or both? Does your application care about the single best
  answer, or is "the right answer was in the top few" good enough?

**Our answer:**

---

# Part 3 — Metric Pitfalls

You always need to make sure that you load your dataset's annotations correctly.
In Lab 4c, the pretrained detector predicts bounding boxes of **whole animals**.
But if you look at the original annotations of the The Oxford-IIIT Pet Dataset,
the bounding boxes mark **only the head of each animal** (see [here](https://www.robots.ox.ac.uk/~vgg/data/pets/)). If you use a pre-trained model that has a different convention 
for the annotations than the ground-truth annotations, you could get evaluation metrics
that look artificially bad. This kind of pitfall is common, and it's much cheaper to catch before you've run your experiments than after.

## 5. What is the equivalent risk for your own project's data and labels?

Think about units, coordinate systems, class-index conventions, how positive/negative is defined,
or anything else where your predictions and your ground truth could technically both be "correct"
by their own convention but not directly comparable.

> Our risk:

## 6. Pitfall checklist

Check anything you still need to verify before you can trust your evaluation numbers:

- [ ] Metric will be computed on a held-out validation/test set, not the training set
- [ ] Ground truth and predictions use the same convention (units, box format, class indices, etc.)
- [ ] Accuracy alone is not used if your classes are imbalanced (see your week 3 worksheet)
- [ ] More than one threshold/IoU level will be inspected, not just a single point estimate
- [ ] Predictions will also be inspected visually, not judged from a single number alone
- [ ] The metric actually reflects what your research question in week 2 is asking about

---

# Part 4 — Qualitative Evaluation Plan

## 7. What will your "green box / red box" equivalent look like?

Lab 4b and 4c both ended with a figure comparing ground truth and prediction side by side, and
Lab 4a with a confusion matrix. Sketch (in words) the figure(s) you plan to show in your report.

| Figure | What it shows | Best case / worst case? |
|---|---|---|
| | | |
| | | |

## 8. What might explain your model's likely failure cases?

You may not know yet — write your best guess now, and revisit after you've actually trained
something. Common causes include class confusion, occlusion, small or rare objects, domain shift
between training and test data, or label noise (see your week 3 data assessment).

> Our guess:

---

# Part 5 — From Metrics to Report Sections

Today's lecture covered the report structure (Introduction → Related Work → Methods → Results →
Discussion) and the Context/Problem/Solution/Findings/Limitations/Conclusions framing. Use what
you decided above to draft placeholder text — it's fine if it's speculative or incomplete.

## 9. Draft one sentence for your (future) Results section

Objective, no interpretation — just what was measured, per the lecture's advice ("describe
objectively what you see – do not discuss results here").

> Example: "Our fine-tuned model achieves 82% top-1 accuracy on the held-out test set, compared
> to 65% for the majority-class baseline."

**Our draft:**

## 10. Draft one sentence for your (future) Discussion section

This is where interpretation belongs: what worked, what didn't, and why.

> Example: "The model performs well on well-lit, centered images but its error rate roughly
> doubles on the subset of images with partial occlusion, suggesting the training data
> under-represents that condition."

**Our draft:**

## 11. Is there a benchmark or published result you should compare against?

Check your week 3 literature search — if your dataset (or a closely related one) has a published
result, name the metric and value here so you have a reference point.

**Our benchmark (or "none found"):**

---

# Part 6 — Evaluation Code Plan

## 12. Which `torchmetrics` classes (or other tools) will you use?

| Metric | Library / class | Notes |
|---|---|---|
| | | |
| | | |
| | | |

## 13. Will you need any custom code, similar to Lab 4c's trimap-to-box conversion?

Describe anything you'll need to write yourself to get your predictions and ground truth into a
comparable format (e.g. converting model output to the same coordinate system or units as your
labels).

> Our answer:

---

# Part 7 — Next Steps

- [ ] Implement the primary metric on your sanity-check baseline
- [ ] Confirm ground truth and prediction formats actually match (Part 3)
- [ ] Draft the figure(s) planned in Part 4, even with placeholder/baseline predictions
- [ ] Research the metric for your task if it wasn't covered by today's notebooks (Part 1)
- [ ] Bring this worksheet and your notebook to office hours if anything is unclear
- [ ] Other: ______________________________

---

**Keep this worksheet — it is the basis for the Methods and Results sections of your final
report.**
