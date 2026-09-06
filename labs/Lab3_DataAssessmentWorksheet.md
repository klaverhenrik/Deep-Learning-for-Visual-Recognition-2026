# Data Assessment Worksheet
### Deep Learning for Visual Recognition — Week 3

**Group members:**

**Project title (from week 2):**

**Date:**

---

## Instructions

You should have completed the week 2 scoping worksheet before this session.
Today's goal is to evaluate your actual dataset against the project you have in mind.

Work through this worksheet together. Run the sanity check functions from the
lab notebook on your data and record what you find.

**Note:** If you still have multiple project ideas, pick one of them for this lab. If you
haven't found a problem or dataset yet, pick one from Kaggle (e.g., https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification). You can search datasets here: https://www.kaggle.com/datasets?search=image&tags=13207-Computer+Vision

**A note on task type:** The lab notebook demonstrates the sanity checks on an
image *classification* dataset. If your project is a different vision task
(detection, segmentation, regression, or something else), some questions
below won't apply exactly as written — each check includes a short note on
how to adapt it. If a question genuinely doesn't apply to your task, write
**N/A** and a one-sentence reason, rather than leaving it blank.

---

## Part 1: Dataset Identity

**1. What dataset are you using?**

| | |
|---|---|
| **Name / source** | |
| **URL or reference** | |
| **How you obtained it** | Downloaded / scraped / self-collected |

**Task type:**
- [ ] Image classification
- [ ] Object detection
- [ ] Semantic / instance segmentation
- [ ] Regression (predicting a continuous value from an image)
- [ ] Other: 

---

**2. Does this dataset match what you described in your week 2 worksheet?**

Compare today's dataset to your week 2 answers on input/output, classes, and data source.

- [ ] Yes — the dataset matches the plan exactly
- [ ] Mostly — minor differences (describe below)
- [ ] No — we changed our dataset (describe why below)

_Notes:_

---

## Part 2: Sanity Check Results

Figure out how to download and load the dataset and run the five sanity
checks from the lab notebook. Record your results here. The notebook's code
is written for classification (`ImageFolder`) — if your task is different,
adapt the underlying logic rather than the code as-is; the notes under each
check below explain what to adapt.

### Check 1: Visual inspection

Identify a few representative image examples (per class if relevant), or describe what you see.

_What do the images look like? Are the labels (if any) obviously correct?_

_Any classes or other things that look ambiguous, mislabelled, os suspecious?_

*If your task doesn't have discrete classes (e.g. detection, segmentation,
regression), inspect a few examples together with their annotations
(boxes / masks / target values) instead, and comment on whether those look
correct.*

---

### Check 2: Class / label distribution

*Classification: fill in the table below as-is, counting images per class.*

*Detection: count **instances** (bounding boxes) per class, not images — one
image can contain several instances of several classes.*

*Segmentation: report the proportion of labelled **pixels** per class instead
of image counts, since one image usually contains multiple classes.*

*Regression (no discrete classes): skip the table and instead report the
distribution of your target value — minimum, maximum, mean, and a rough sense
of shape (e.g. from a histogram).*

| Class name | Training count | Validation count | Test count (if available) |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| **TOTAL** | | | |

**Imbalance ratio** (max class / min class), or N/A for regression: _______________

**Is there a class imbalance problem?**
- [ ] No — classes are roughly balanced (ratio < 3x)
- [ ] Mild imbalance (ratio 3–10x) — may need weighted loss
- [ ] Severe imbalance (ratio > 10x) — needs resampling or weighted loss
- [ ] N/A — task has no discrete classes (describe target distribution above instead)

---

### Check 3: Image size distribution

| | Width (pixels) | Height (pixels) |
|---|---|---|
| **Minimum** | | |
| **Maximum** | | |
| **Median** | | |

**Are the sizes consistent?**
- [ ] Yes — images are roughly the same size
- [ ] No — large variation (min-to-max ratio > 10x)

_Notes on size variation:_

*This check applies the same way regardless of task type — it looks at the
raw images, not the labels.*

---

### Check 4: Tensor statistics after normalisation

| Channel | Mean (should be ≈ 0) | Std (should be ≈ 1) |
|---|---|---|
| R | | |
| G | | |
| B | | |

**Does normalisation look correct?**
- [ ] Yes — mean near 0, std near 1
- [ ] No — something looks wrong (describe below)

_Notes:_

*This check also applies the same way regardless of task type.*

---

### Check 5: Train / validation split

| | Training | Validation | Val % |
|---|---|---|---|
| **Total** | | | |

**Is the split reasonable?**
- [ ] Yes — validation is 15–25% of total
- [ ] Validation is too small (< 10%)
- [ ] Validation is too large (> 40%)

**Are any corrupt images found?**
- [ ] No corrupt images
- [ ] Yes — number found: _______

*If your task has no discrete classes, just report the overall split
percentages above — you don't need a per-class breakdown.*

---

## Part 3: Data Quality Assessment

**3. What is the overall quality of your data?**

Rate each dimension:

| Dimension | Rating | Notes |
|---|---|---|
| **Label accuracy** | Good / Acceptable / Poor | |
| **Image quality** | Good / Acceptable / Poor | |
| **Dataset size** | Sufficient / Marginal / Insufficient | |
| **Class balance** (or target distribution, if no discrete classes) | Good / Acceptable / Poor | |
| **Consistency** (same conditions across images) | Good / Acceptable / Poor | |

---

**4. What are the biggest data problems you found?**

List up to three, most serious first:

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

---

**5. How will you address these problems?**

| Problem | Planned solution |
|---|---|
| | |
| | |
| | |

---

## Part 4: Feasibility Update

**6. Given what you now know about your data, is your project still feasible?**

- [ ] Yes — the data looks good, no major concerns
- [ ] Yes with adjustments — describe changes below
- [ ] Uncertain — need to investigate further (what are you waiting on?)
- [ ] No — we need a different dataset or different project

_Notes:_

---

**7. Has your project scope changed since week 2?**

Compare to your week 2 scoping worksheet. Describe any changes:

---

**8. What do you still not know about your data?**

List open questions that you need to resolve before the week 4 proposal:

1. _______________________________________________

2. _______________________________________________

---

## Part 5: Literature

Your proposal must include 1–3 references to relevant research papers.
Use this section to record what you have found today.

**9. What papers have you found that are relevant to your problem?**

| Paper title | Authors / year | Their result | How it relates to your project |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

---

**10. What is the state of the art on your problem?**
What is the best published result on your dataset or a closely related one?
This gives you a reference point for your own results.

_Write your answer here (if known):_

---

**11. What makes your approach different from prior work?**
You do not need to be doing something novel — but you should be able to
explain how your project relates to what has already been done.

_Write your answer here:_

---

## Part 6: Next Steps

**12. What do you need to do before handing in the final project proposal?**

| Task | Who |
|---|---|
| | |
| | |
| | |

---


---
