# Project Scoping Worksheet

**Week 2 — From Project Idea to Testable Investigation**

**Group members:**  
**Date:**  

---
## Instructions
Fill in this worksheet alone or together as a group during today's lab session.
If you have multiple project ideas, you can work through the sheet multiple times.
There are no right or wrong answers — the goal is to think through your
project idea carefully before committing to it. The TAs will come around
during the session to give you input and feedback.

## Purpose of this exercise

By the end of the lab, your group should have a **provisional, feasible project plan** containing:

- a visual recognition task and dataset;
- one primary research question;
- a simple, credible baseline;
- at least one controlled experiment;
- an evaluation plan; and
- a minimum viable project and fallback plan.

Your answers may change after you inspect the data and read related work. Uncertainty is expected at this stage — record it rather than hiding it.

Ask a TA for feedback before the end of the session. You will refine this worksheet into the final project proposal during Week 4.

**Suggested lab flow:** Complete Parts 1–3 first, discuss them briefly with a TA, and then draft Parts 4–6. Part 7 may be completed after the lab if time is limited.

---

# Part 1 — Problem and Task

## 1. What problem do you want to investigate?

Explain the problem in plain language. Avoid model names for now.

> Example: “Given an image of a plant leaf, predict which disease is present.”

**Our problem:**

## 2. Why is this problem interesting or useful?

Who might care about the result? What makes the problem non-trivial?

## 3. What are the inputs and outputs?

| | Description | Example |
|---|---|---|
| **Input** | | |
| **Output/target** | | |

## 4. What is the primary task type?

- [ ] Image classification
- [ ] Object detection
- [ ] Semantic or instance segmentation
- [ ] Image generation
- [ ] Regression
- [ ] Retrieval, matching, or verification
- [ ] Vision-language or multimodal understanding
- [ ] Other: ______________________________

If your project combines several tasks, identify the **primary task that you will evaluate**:

---

# Part 2 — Research Question and Hypothesis

The project should be an investigation, not only an application of a pretrained model. A useful project asks what changes, why it may help, and how the claim will be tested.

## 5. What is your provisional research question?

Try to express it as a comparison:

> “How does **X** affect **Y** under **Z conditions**, compared with **baseline B**?”

Possible factors include augmentation, loss function, amount of training data, fine-tuning strategy, model capacity, class imbalance, label noise, domain shift, or another justified method.

**Our primary research question:**

## 6. What is your initial hypothesis?

State both the expected result and your reasoning. A hypothesis may turn out to be wrong; a well-designed negative result is still useful.

> “We expect ___ because ___.”

---

# Part 3 — Data Feasibility

## 7. What data will you use?

- [ ] Existing public dataset
- [ ] Data supplied by a course partner or researcher
- [ ] Data collected by the group
- [ ] Data scraped from the web
- [ ] Synthetic or generated data
- [ ] Other: ______________________________

**Dataset name/source/link:**

**Can you access and legally use it for the project?**  Yes / No / Unsure

## 8. What counts as one independent example?

For example: one image, one video, one patient, one specimen, or one scene. Note whether several files can come from the same source.

For example:
- 10,000 unrelated photographs → approximately 10,000 independent examples.
- 10,000 video frames from 100 videos → closer to 100 independent sources.
- 5,000 medical images from 500 patients → 500 independent patients.
- Multiple augmented versions of 1,000 images → still only 1,000 original examples.
- Several photos of each of 200 objects → 200 independent objects.

This matters because closely related images must stay in the same train/validation/test split. Otherwise, the model may see nearly identical information during training and testing, producing overly optimistic results due to data leakage.

## 9. What targets or annotations are available?

Examples include class labels, bounding boxes, masks, captions, paired images, or continuous measurements.

| Question | Answer |
|---|---|
| Annotation/target type | |
| Approximate number of independent examples | |
| Number of classes or target categories, if applicable | |
| Known imbalance or rare cases | |
| Annotation quality or missing labels | |

## 10. How should the data be split?

Propose a training/validation/test split. Explain what must be grouped to prevent leakage — for example, images from the same patient, video, object, location, or near-duplicate source must not appear across splits.

| Split | Approximate size | Purpose |
|---|---:|---|
| Training | | Fit model parameters |
| Validation | | Select models and settings |
| Test | | Final evaluation only |

**Grouping or leakage constraints:**

## 11. Inspect several examples from the dataset

Do not rely only on the dataset description. Record at least three observations that could affect learning or evaluation.

| Observation | Why it may matter |
|---|---|
| 1. | |
| 2. | |
| 3. | |

Potential issues to consider: duplicates, label noise, class imbalance, image quality, shortcuts or background cues, domain shift, and unrepresentative data.

---

# Part 4 — Baseline and Experimental Plan

## 12. What is the simplest credible baseline?

The transfer-learning example from today's lab is a reasonable starting pattern for many projects. Other tasks may require a pretrained detector, segmenter, generator, or vision-language model.

Using a pretrained or plug-and-play model is acceptable as a **starting baseline**. It is not, by itself, a complete course project. Your group must own the question, adaptation or training procedure, experiment design, evaluation, and interpretation.

**Baseline model or method:**

**What will be trained or fine-tuned?**

**Why is this the simplest baseline that can answer your question?**

## 13. Plan a small experiment ladder

Each experiment should change one main conceptual factor while keeping other important choices fixed. You do not need four experiments; complete the rows that are currently realistic.

| Experiment | Main change from baseline | Question answered | Evidence to collect |
|---|---|---|---|
| **E0: Baseline** | — | How well does the basic approach work? | |
| **E1** | | | |
| **E2** | | | |
| **E3 / extension** | | | |

**What should remain fixed so the comparison is fair?**

Examples: data split, evaluation code, number of runs, training budget, preprocessing, and model-selection procedure.

---

# Part 5 — Evaluation

## 14. How will you evaluate the project?

You will learn in detail about evaluation methods and metrics in Week 4. For now, do your best to figure out which methods/metrics are appropriate to your task. Accuracy alone may hide poor performance on rare classes; generated outputs may require both quantitative and qualitative evaluation.

| Evaluation component | Choice | Why is it appropriate? |
|---|---|---|
| Primary metric | | |
| Secondary metric(s) | | |
| Per-class, subgroup, or robustness analysis | | |
| Qualitative examples or error analysis | | |

## 15. What evidence would answer the research question?

Do not define project success only as reaching a particular score. Explain what comparison, pattern, or analysis would let you draw a credible conclusion — even if the hypothesis is not supported.

**When will you use the test set?**

---

# Part 6 — Scope, Resources, and Risk

## 16. Define the minimum viable project

What is the smallest complete investigation your group can finish and analyse by the deadline?

It should include a working baseline, one meaningful controlled comparison, suitable evaluation, and error analysis.

## 17. Optional extension

If the minimum project succeeds early, what research-like method, additional hypothesis, robustness test, or broader analysis could you add?

## 18. Feasibility and resources
Some of these questions are hard to answer at this stage. You are encouraged to start experimenting already now.

| Question | Answer |
|---|---|
| Do you need more compute that what Google Colab provides? | |
| Rough duration of one training run | |
| How many runs may be needed? | |
| Is annotation tooling needed, and if yes, is it available? | |
| What must you verify this week? | |

## 19. What is the biggest risk?

- [ ] Data cannot be accessed or used
- [ ] Too little data or poor annotations
- [ ] Data leakage or invalid evaluation
- [ ] Training is too computationally expensive
- [ ] Baseline implementation may not work
- [ ] Research question is too broad or vague
- [ ] Group lacks required domain knowledge
- [ ] Other: ______________________________

**Risk:**

**Fallback or mitigation:**

---

# Part 7 — Related Work and Next Steps

## 20. Find 1–3 closely related projects or papers

Use Google Scholar or references associated with the dataset. At this stage, focus on discovering reasonable baselines, evaluation practices, and realistic project scope.

| Title and link | Task/data | Method and baseline | Evaluation | What we can learn or test differently |
|---|---|---|---|---|
| 1. | | | | |
| 2. | | | | |
| 3. | | | | |

## 21. Open questions for the TA

List up to three questions that block or materially affect your project plan.

1.  
2.  
3.  

---

# TA Feedback

## Quick scope check

| Item | Too vague / risky | Plausible but needs work | Clear and feasible |
|---|:---:|:---:|:---:|
| Task and output | ☐ | ☐ | ☐ |
| Research question and hypothesis | ☐ | ☐ | ☐ |
| Data access and split validity | ☐ | ☐ | ☐ |
| Baseline | ☐ | ☐ | ☐ |
| Controlled experiment | ☐ | ☐ | ☐ |
| Evaluation plan | ☐ | ☐ | ☐ |
| Minimum viable scope and fallback | ☐ | ☐ | ☐ |

## Most important feedback

## Actions before the next project session

- [ ] Confirm data access, licence, and approximate size
- [ ] Inspect examples and check for duplicates or leakage risks
- [ ] Run or identify a feasible baseline
- [ ] Refine the research question and hypothesis
- [ ] Read at least one closely related paper or project
- [ ] Estimate training time on a small subset
- [ ] Other: ______________________________

---

**Keep this worksheet. It is the starting point for your project proposal, not a final commitment to every choice.**
