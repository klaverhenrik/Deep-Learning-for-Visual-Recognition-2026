# Proposal Slide Template
### Deep Learning for Visual Recognition — Week 4

**Instructions:** Prepare 1–2 slides using this structure.
You will walk an instructor or TA through them during today's session.
Keep it concise — you have 5 minutes maximum.

---

## Slide 1: The Project

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PROJECT TITLE                                                  │
│  ─────────────────────────────────────────────────────────     │
│                                                                 │
│  PROBLEM (1–2 sentences)                                        │
│  What are you trying to do? Who would benefit?                  │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │   INPUT              │   │  OUTPUT                      │   │
│  │                      │   │                              │   │
│  │  [example image or   │──▶│  [label / mask / generated   │   │
│  │   description]       │   │   image / bounding box]      │   │
│  └──────────────────────┘   └──────────────────────────────┘   │
│                                                                 │
│  TASK TYPE                                                      │
│  □ Classification  □ Detection  □ Segmentation                  │
│  □ Generation      □ Other: ___________________                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Fill in:**
- Project title
- 1–2 sentence problem description
- Concrete example of one input and its expected output
- Task type

---

## Slide 2: Data, Approach, and Success

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DATA                                                           │
│  ─────────────────────────────────────────────────────────     │
│  Source:      ____________________________________________      │
│  Size:        _______ images  /  _______ classes               │
│  Split:       _______ train  /  _______ val  /  _______ test   │
│  Known issues: ___________________________________________      │
│                                                                 │
│  APPROACH                                                       │
│  ─────────────────────────────────────────────────────────     │
│  Baseline:    ____________________________________________      │
│  Main model:  ____________________________________________      │
│  Why this approach (not a simpler one):                         │
│               ____________________________________________      │
│                                                                 │
│  PLANNED ITERATIONS                                             │
│  ─────────────────────────────────────────────────────────     │
│  Expected challenge:  _____________________________________     │
│  Experiment 1:        _____________________________________     │
│  Experiment 2:        _____________________________________     │
│                                                                 │
│  SUCCESS CRITERION                                              │
│  ─────────────────────────────────────────────────────────     │
│  Metric:      ____________________________________________      │
│  Target:      ____________________________________________      │
│                                                                 │
│  RELATED WORK (1–3 papers)                                      │
│  ─────────────────────────────────────────────────────────     │
│  1. ______________________________________________________      │
│  2. ______________________________________________________      │
│  3. ______________________________________________________      │
│  How your approach differs: ______________________________      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Fill in:**
- Dataset source, size, split, and any known issues
- What your baseline is and what your main approach will be
- A concrete measurable success criterion
- The biggest risk and how you plan to handle it

---

## Checklist before the session

Make sure you can answer all of these when talking to an instructor or TA:

**The problem:**
- [ ] Can you explain your problem in one sentence to someone outside your field?
- [ ] Can you give a concrete example of one input and the correct output?

**The data:**
- [ ] Do you know exactly where your data is coming from?
- [ ] Have you seen the data? Have you run the sanity checks from week 3?
- [ ] Do you know how many images you have per class?

**The approach:**
- [ ] Do you know what your baseline will be?
- [ ] Do you know what pretrained model you will start from, if any?

**Success:**
- [ ] Have you defined a concrete, measurable success criterion?
- [ ] Do you know what metric to use and why?

**Risk:**
- [ ] Have you identified the biggest thing that could go wrong?

If you cannot answer any of these, that is what the feedback session is for.

---

## What makes a good proposal

The instructors and TAs will be looking for:

**Clarity** — Is the problem well-defined? Is the input/output relationship unambiguous?

**Feasibility** — Is the dataset real and accessible? Is the scope achievable in the time available?

**Appropriate ambition** — Is the project interesting and challenging, but not so ambitious it is impossible?

**Honest risk assessment** — Does the group understand what could go wrong?

Common problems to avoid:
- Vague problem statements ("we want to use deep learning to analyse images")
- No concrete dataset yet ("we will collect data later")
- Success criteria that cannot be measured ("the model should work well")
- Scope that is too large for one semester ("we will build a real-time video analysis system")
