# Writing a Related Work Section
### Deep Learning for Visual Recognition — Week 11

This guide is for the second half of today's session, after you have
worked on your results and figures. If your related work section is
already written and solid, use the time for project work instead.

---

## Why related work matters

The related work section is where you demonstrate that you understand
your problem in the context of what the field has already done.
It is not a literature dump — it is an argument.

The argument is: *here is what others have tried, here is what they
achieved, and here is why my approach is a reasonable next step.*

A strong related work section shows the reader:
1. You have done a proper literature search
2. You understand the key prior work, not just its existence
3. You can explain how your approach relates to — and differs from — prior work

A weak related work section is a list of papers with one-sentence summaries
that have no connection to what you actually did.

---

## Structure

A related work section for a 6–8 page course project should be
**roughly one page** — enough to cover 3–5 papers with real discussion.

A useful structure:

**1. Frame the problem space (1–2 sentences)**
What is the general area your project belongs to?
*'Object detection in aerial images has received significant attention
in recent years, driven by the availability of drone imagery and
satellite datasets.'*

**2. Discuss the most relevant prior work (the bulk of the section)**
For each paper: what did they do, what dataset, what result, what
is relevant to your project. See the templates below.

**3. Position your own work (1–2 sentences)**
How does your approach differ? What gap does it address, even if modestly?
*'Unlike Chen et al. who use a two-stage detector, we investigate
whether a simpler single-stage approach is sufficient for our
more constrained domain.'*

---

## Templates for discussing a paper

Use these as starting points, not formulae. Vary the structure to
avoid all your paragraphs sounding identical.

**When the paper uses the same dataset:**
> *[Author et al., year] established a benchmark on [dataset],
> achieving [result] using [method]. Their approach [key aspect].
> We use the same evaluation protocol to enable direct comparison.*

**When the paper uses a different but related dataset:**
> *[Author et al., year] addressed a similar problem of [task]
> on [dataset], reporting [result]. While their data differs from ours
> in [key way], their finding that [key insight] informs our choice of [your decision].*

**When the paper proposes the architecture you are using:**
> *[Author et al., year] introduced [architecture], which [how it works].
> We use this as our backbone because [reason — connects to your problem].*

**When the paper shows something negative that motivates your approach:**
> *[Author et al., year] showed that [approach] performs poorly when
> [condition], achieving only [result]. Our dataset exhibits [condition],
> which motivated us to [your choice] instead.*

---

## What not to do

**Do not summarise every paper you found.**
Three papers discussed well is better than eight papers mentioned superficially.

**Do not just list methods without connecting them to your work.**
Every paper you cite should be there for a reason that you state explicitly.

**Do not cite papers you have not read.**
If you only read the abstract, say what you learned from the abstract.
If you read the full paper, you will write about it better.

**Do not use LLMs to write your related work.**
They will hallucinate results and citations. This is plagiarism even
if you did not intend it. Every number and claim must come from the
actual paper.

**Do not copy sentences from the papers.**
Paraphrase. You are demonstrating your understanding, not reproducing theirs.

---

## The "so what" test

After writing each paragraph about a paper, ask: *so what?*
Why did you include this? If you cannot answer that clearly,
the paragraph needs either a stronger connection to your own work,
or it should be cut.

The reader should finish your related work section understanding:
- What the current state of the art is on your problem
- What the main approaches are and their trade-offs
- Where your project sits in that landscape

---

## Connecting related work to your methods and results

Related work is not isolated from the rest of your report.
Strong reports cross-reference it:

- **In your Methods section:** *'Following [Author et al.], we use
  [technique] because [reason].'*
- **In your Results section:** *'Our result of X% compares to
  [Author et al.]'s Y% on a similar task, though direct comparison
  is limited by [difference in setup].'*
- **In your Discussion:** *'[Author et al.] found [insight] in a
  different domain; our results suggest this generalises to [your domain].'*

This kind of cross-referencing is what the SOLO taxonomy means by
*relating and applying* concepts — and it is what distinguishes a
strong report from one that merely describes.

---

## Today's task

If your related work section is not yet written, spend 20–30 minutes
on it now using this guide.

Work through these steps:

1. Gather the 2–3 papers you cited in your proposal
2. Re-read the abstract and results section of each one
3. For each paper, write 2–4 sentences: what they did, what they got,
   what is relevant to you
4. Add one sentence at the end of the section positioning your own work
5. Apply the "so what" test to every paragraph

A TA can review your related work section today if you want feedback
before submission.
