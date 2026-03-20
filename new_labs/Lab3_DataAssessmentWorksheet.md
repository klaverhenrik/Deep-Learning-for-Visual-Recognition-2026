# Data Assessment Worksheet
### Deep Learning for Visual Recognition — Week 3

**Group members:** _______________________________________________

**Project title (from week 2):** _______________________________________________

**Date:** _______________________________________________

---

## Instructions

You should have completed the week 2 scoping worksheet before this session.
Today's goal is to evaluate your actual dataset against the project you proposed.

Work through this worksheet together. Run the sanity check functions from the
lab notebook on your data and record what you find. A TA will review your
findings before the end of the session.

---

## Part 1: Dataset Identity

**1. What dataset are you using?**

| | |
|---|---|
| **Name / source** | |
| **URL or reference** | |
| **Licence** | |
| **How you obtained it** | Downloaded / scraped / self-collected |

---

**2. Does this dataset match what you described in your week 2 worksheet?**

Compare today's dataset to your week 2 answers on input/output, classes, and data source.

- [ ] Yes — the dataset matches the plan exactly
- [ ] Mostly — minor differences (describe below)
- [ ] No — we changed our dataset (describe why below)

_Notes:_

&nbsp;

&nbsp;

---

## Part 2: Sanity Check Results

Run the five sanity checks from the lab notebook and record your results here.

### Check 1: Visual inspection

Paste or sketch a few representative examples per class, or describe what you see.

_What do the images look like? Are the labels obviously correct?_

&nbsp;

&nbsp;

_Any classes that look ambiguous or mislabelled?_

&nbsp;

---

### Check 2: Class distribution

| Class name | Training images | Validation images | Test images (if available) |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| **TOTAL** | | | |

**Imbalance ratio** (max class / min class): _______________

**Is there a class imbalance problem?**
- [ ] No — classes are roughly balanced (ratio < 3x)
- [ ] Mild imbalance (ratio 3–10x) — may need weighted loss
- [ ] Severe imbalance (ratio > 10x) — needs resampling or weighted loss

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

&nbsp;

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

&nbsp;

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

---

## Part 3: Data Quality Assessment

**3. What is the overall quality of your data?**

Rate each dimension:

| Dimension | Rating | Notes |
|---|---|---|
| **Label accuracy** | Good / Acceptable / Poor | |
| **Image quality** | Good / Acceptable / Poor | |
| **Dataset size** | Sufficient / Marginal / Insufficient | |
| **Class balance** | Good / Acceptable / Poor | |
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

&nbsp;

&nbsp;

---

**7. Has your project scope changed since week 2?**

Compare to your week 2 scoping worksheet. Describe any changes:

&nbsp;

&nbsp;

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

&nbsp;

---

**11. What makes your approach different from prior work?**
You do not need to be doing something novel — but you should be able to
explain how your project relates to what has already been done.

_Write your answer here:_

&nbsp;

&nbsp;

---

## Part 6: Next Steps

**12. What do you need to do before the week 4 proposal workshop?**

| Task | Who | Deadline |
|---|---|---|
| | | Week 4 lab |
| | | Week 4 lab |
| | | Week 4 lab |

---

## TA Feedback

_For TA use during the session:_

**Is the dataset suitable for the project?** [ ] Yes  [ ] Needs work  [ ] Unclear

**Are the data problems manageable?** [ ] Yes  [ ] Risky  [ ] Serious concern

**Has the group found relevant literature?** [ ] Yes  [ ] Needs work  [ ] Not yet

**Is the group ready for the proposal?** [ ] Yes  [ ] Nearly  [ ] Needs more prep

**Key feedback:**

&nbsp;

&nbsp;

&nbsp;

**Action items before week 4:**

1. _______________________________________________

2. _______________________________________________

---

*Bring both worksheets (week 2 and week 3) to the week 4 proposal workshop.*
*Your proposal should incorporate everything you have learned from them.*
