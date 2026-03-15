# Project Scoping Worksheet
### Deep Learning for Visual Recognition — Week 2

**Group members:** _______________________________________________

**Date:** _______________________________________________

---

## Instructions

Fill in this worksheet together as a group during today's lab session.
There are no right or wrong answers — the goal is to think through your
project idea carefully before committing to it. A TA will come around
to give feedback before the end of the session.

You will refine this further in weeks 3 and 4, leading to your
formal project proposal.

---

## Part 1: The Problem

**1. What is the problem you want to solve?**
Describe it in plain language — no technical jargon. What would someone
who knows nothing about deep learning understand from your description?

_Write your answer here (3–5 sentences):_

&nbsp;

&nbsp;

&nbsp;

---

**2. Why is this problem interesting or important?**
Who would benefit from a solution? What is currently done instead?

_Write your answer here:_

&nbsp;

&nbsp;

---

**3. What is the input and what is the output?**
Be specific. What does one example look like?

| | Description | Example |
|---|---|---|
| **Input** | | |
| **Output** | | |

---

**4. What type of problem is it?**
Tick all that apply:

- [ ] Image classification (one label per image)
- [ ] Object detection (bounding boxes + labels)
- [ ] Image segmentation (pixel-level labels)
- [ ] Image generation
- [ ] Something else: _______________________________________________

---

## Part 2: The Data

**5. Where will your data come from?**
Tick all that apply and describe:

- [ ] Existing public dataset — name/link: _______________________________________________
- [ ] We will collect it ourselves — how: _______________________________________________
- [ ] We will scrape it from the web — source: _______________________________________________
- [ ] Other: _______________________________________________

---

**6. How much data do you expect to have?**

| Split | Approx. number of images |
|---|---|
| Training set | |
| Validation set | |
| Test set | |
| **Total** | |

If you are unsure, write your best estimate and note what it depends on.

&nbsp;

---

**7. What are the classes / categories?**
List them and give the expected number of examples per class.
Flag any class imbalance you already know about.

| Class name | Expected number of examples | Notes |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

---

**8. What are the potential data problems?**
Think about: quality, labelling noise, class imbalance, domain shift,
privacy or ethical concerns, licensing restrictions.

_Write your answer here:_

&nbsp;

&nbsp;

&nbsp;

---

## Part 3: The Approach

**9. What is your planned approach?**
You do not need to have this fully worked out yet.
Which of the following best describes your starting point?

- [ ] Fine-tune a pretrained CNN (ResNet, EfficientNet, ViT...)
- [ ] Train from scratch
- [ ] Use a pretrained model with no fine-tuning (zero-shot / feature extraction)
- [ ] Build on top of an existing architecture (e.g. YOLO, Mask R-CNN, Stable Diffusion)
- [ ] Not sure yet

What backbone / architecture are you considering, if any?

_Write your answer here:_

&nbsp;

---

**10. What is your baseline?**
A baseline is the simplest thing you could try first.
It gives you a reference point to improve on, and it often reveals
problems with your data or setup before you invest in complex solutions.

What is the simplest baseline you could run in your first week of implementation?

_Write your answer here:_

&nbsp;

&nbsp;

---

## Part 4: Success and Risk

**11. What does success look like?**
Define a concrete, measurable goal. Avoid vague targets like "high accuracy".

| Target | Value | Justification |
|---|---|---|
| Primary metric (e.g. accuracy, mAP, FID...) | | |
| Minimum acceptable result | | |
| Stretch goal | | |

---

**12. What is the biggest risk to your project?**
What is the single thing most likely to prevent you from getting a good result?

- [ ] Not enough data
- [ ] Data is hard to collect or label
- [ ] The problem may be too hard for the time available
- [ ] We are not sure the approach will work
- [ ] Compute constraints
- [ ] Something else: _______________________________________________

How do you plan to mitigate this risk?

_Write your answer here:_

&nbsp;

&nbsp;

---

## Part 5: Open Questions

**13. What do you not know yet that you need to find out?**
List up to three open questions that you need to answer before you can
commit to this project.

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

---

## TA Feedback

_For TA use during the session:_

**Is the problem well-scoped?** [ ] Yes  [ ] Needs work  [ ] Too broad  [ ] Too narrow

**Is the data plan realistic?** [ ] Yes  [ ] Needs work  [ ] Unclear

**Is the approach appropriate?** [ ] Yes  [ ] Needs work  [ ] Unclear

**Key feedback:**

&nbsp;

&nbsp;

&nbsp;

**Action items before week 3:**

1. _______________________________________________

2. _______________________________________________

---

*Bring this worksheet to the week 3 lab. You will use it when evaluating your dataset.*
