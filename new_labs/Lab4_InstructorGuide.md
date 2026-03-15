# Week 4 Proposal Workshop — Instructor & TA Guide

## Overview

**Format:** Science fair / rotating feedback  
**Groups:** ~50  
**Instructors/TAs:** 3  
**Session length:** 3 hours  
**Groups per instructor:** ~17  
**Time per group:** ~10 minutes  

The proposal is submitted *after* this session. The feedback groups receive
today should directly improve their written proposal.

---

## Session Timeline

| Time | What is happening |
|---|---|
| 0:00–0:10 | Instructor intro — explain the format, what to expect |
| 0:10–0:20 | Groups set up, open notebooks, finalise slides |
| 0:20–2:40 | Rotating feedback rounds (see below) |
| 2:40–3:00 | Wrap-up; any groups with unresolved issues get a final pass |

---

## Setup (before session)

Divide the 50 groups into three lists of ~17 — one per instructor/TA.
Assign each list a section of the room so groups know who is coming to them.

Share the group lists with each other before the session starts.
Each instructor/TA works through their own list independently.

---

## The Feedback Round

Each group gets approximately **10 minutes**:

| Minutes | What to do |
|---|---|
| 0–2 | Group walks you through their slides (do not interrupt) |
| 2–6 | Ask questions, probe weak points (see question bank below) |
| 6–9 | Give concrete feedback — what is strong, what needs work |
| 9–10 | Write 1–2 action items in their week 2 or 3 worksheet |

**Keep to time.** If you fall behind, shorten the question phase.
It is better to give every group brief feedback than to give half the groups
thorough feedback and miss the rest.

---

## Question Bank

Use these to probe proposals that seem underspecified. You do not need to
ask all of them — pick the ones most relevant to the group's situation.

**On the problem:**
- "Can you give me a concrete example? Show me one image and tell me what the correct output is."
- "Who would use this, and what would they do differently if the model worked?"
- "How is this different from [similar existing tool/dataset]?"

**On the data:**
- "Have you actually seen the data yet, or is this still theoretical?"
- "How did you verify the labels are correct?"
- "What happens if your data source is unavailable or too expensive to collect?"
- "How many images per class do you have? Is that enough?"

**On the approach:**
- "What is your baseline, and when will you have a baseline result?"
- "Why did you choose this architecture over alternatives?"
- "What will you do if the baseline is already good enough?"

**On success:**
- "How will you know when you are done?"
- "What does [their metric] of [their target] actually mean in practice — is that good enough for real use?"
- "What is the state of the art on this problem, if known?"

**On risk:**
- "What is the most likely reason this project fails? Have you thought about that?"
- "If you cannot get your planned dataset, what is plan B?"

---

## What to Look For

**Green flags — project is in good shape:**
- Concrete input/output example they can show you
- Dataset already downloaded and sanity-checked
- Realistic scope for one semester
- Clear baseline plan
- Honest about what could go wrong

**Yellow flags — needs attention but fixable:**
- Dataset exists but not yet downloaded or inspected
- Success criterion is vague (push them to make it concrete)
- Approach is overly ambitious (help them scope down to an MVP)
- Risk is identified but mitigation is weak

**Red flags — needs significant rethinking:**
- No concrete dataset ("we will collect data ourselves" with no plan)
- Problem statement is not a visual recognition problem
- Scope is clearly too large (e.g. "we will build a full autonomous system")
- Group has not worked together at all and has no shared understanding

For red flag cases, be direct but constructive. Suggest a concrete pivot
rather than just identifying the problem. If a group is seriously off track,
flag it to the course instructor so they can follow up before the proposal deadline.

---

## Feedback Phrases That Work

**When the problem is too vague:**
"Right now I could not build this from your description. Can you tell me
exactly what one training example looks like — one image and one label?"

**When the dataset is uncertain:**
"The proposal cannot be evaluated without knowing what data you have.
Before you submit, make sure you have found a concrete dataset and
confirmed you can access it."

**When the scope is too large:**
"This is a great long-term vision. For the course project, what is the
smallest version of this that would still be interesting and publishable?
Start there."

**When the success criterion is vague:**
"'Good performance' is not a target you can aim at. What number on what
metric would make you happy? What would make you disappointed?"

**When the group is well-prepared:**
"This is in good shape. The one thing I would push on is [X].
Other than that, you are ready to write this up."

---

## After the Session

Groups submit their proposals after the lab. The proposal template
(shared separately) asks for:

1. Problem statement
2. Dataset description
3. Planned approach and baseline
4. Success criteria and evaluation plan
5. Risk assessment

Your feedback today should help them strengthen all five sections.

If you want to flag a group for follow-up (e.g. still very uncertain about
their dataset), note it during the session and share with the course instructor
after the session ends.
