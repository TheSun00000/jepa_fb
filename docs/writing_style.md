# Writing style for the paper (distilled from the Offline QD paper and the user's corrections)

Reference: "Offline Quality-Diversity" (ICLR 2027 submission, Section 5 and Appendix E), given by the user on 2026-11-19 as the model for the experiments section, abstract and takeaways. Use this file when writing or editing any section of `paper/main.tex`.

## Core principle
- A table or a curve is evidence, not the message. Every result is written as a claim, a number against a reference, a reason, and a caveat. Never write a paragraph that only lists numbers.
- Beating a method is evidence for a question, not the question. Questions are scientific ("what is the effect of shaping the latent with FB?"), never "does X beat Y?".

## Experiments section template
1. **Section opening.** State the questions the section answers, as a short list (Offline QD: "Can any of the three approaches recover ... and how close do they get to online QD?"). Say which subsection answers which question.
2. **One subsection per experiment.** Each one has the same four parts, in this order:
   - Intro paragraph, no title: the motivation of the experiment, what we want to establish. One to three sentences.
   - `\paragraph{Experiment.}` The setup: data, task protocol, baselines (as a list, one line each, with what each one is for), metric, number of seeds, what is reported (final checkpoint, mean and 95% CI over seeds).
   - `\paragraph{Results.}` See below.
   - A `bluebox` takeaway. See below.
3. Global setup (environments, datasets, metrics) can sit in a short block before the first subsection.

## Results paragraphs
- **Start with the claim in words**, then support it. Offline QD: "offline QD is tractable across most of the domains we tested", then the numbers.
- **Every number is tied to a reference.** "96% of the ME ceiling", "76% of ME (3221) and surpassing online QDAC (2134)", "nearly 30x". Percentages of an upper bound, ratios, differences in points. A bare score is not a result.
- **Group by regime or by question**, not row by row. Each group gets its own claim.
- **Give the mechanism after the number.** Why does this happen? Tie it to a design property of the method ("this follows from the guarantee of SF, which holds for additive descriptors"). If the explanation is not tested, mark it: "we hypothesize". Do not state an untested mechanism as a fact.
- **State losses plainly and with a reason.** "Though it remains behind online methods there." Honest losses make the wins credible.
- **Answer the obvious objection before a reviewer asks it**, with a control or a pointer to the appendix. Example: "this comparison gives offline methods access to data that online methods do not have; giving the same data to online methods does not close the gap (Appendix E.2)".
- **Use controls to rule out the alternative explanation** of a gain.
- **Match the claim to the evidence.** If the confidence intervals overlap, say so and rely on the consistent sign across environments, or say "higher mean", not "outperforms". Do not extrapolate beyond the tested range.
- Point to deeper analysis instead of repeating it ("we discuss the causes in Section 5.2").

## Takeaway (bluebox)
- It is the insight, not a summary of the numbers: one to three sentences with the claim, its size, and the reason.
- It answers the question the experiment opened with (if the intro said "how close do they get", the takeaway says how close).
- When results depend on conditions, state the condition: "works when X, fails when Y, because Z". For several methods use bullets with a check mark for when it works and a cross for when it fails, each with its cause.
- It may state what the bottleneck really is ("the main bottleneck is discovering promising regions, not improving the local search").
- No new numbers that the Results did not introduce, and no vague closers ("matters most on cheetah").

## Tables and figures
- Table caption: bold title sentence, then what is shown (mean and std or CI over seeds, number of seeds, what bold and colour mean).
- Std/CI in the table: `\std{..}` ($\pm$, scriptsize). Bold the best among the comparable methods, not the upper bound.
- Reference methods (upper bounds, online methods) are shown but separated from the compared methods.
- Figures: clean, blue template, small footprint, no tall figures that dominate a page. Place the float in the source before the text that discusses it so it lands on the right page (wide floats go to the next page).

## Abstract template (Offline QD)
One paragraph in this order: (1) the field and its limitation in two sentences, (2) "in this paper, we ..." the goal, (3) "we introduce/propose" the method in one or two sentences, (4) "Experiments show ..." with two or three quantified results, (5) one extra finding, (6) "our results suggest ..." the implication. About 200 words.

## Language rules (user's corrections)
- Simple words. No "scarce", "plentiful", "non-trivial"-style vocabulary: "few labels", "more labels".
- Do not write sentences the reader already assumes (for a zero-shot evaluation, do not say the tasks were unseen at pre-training; do not say a task is only useful if the agent can act on it).
- Name the method on first use in a section ("our method FBWM, which trains ... on a shared latent"). The method name is FBWM, written with the `\method{}` macro, never lowercase.
- Questions are short and plain: "How is FBWM affected by the number of labelled states that define the task?".
- Do not quote the per-task adaptation time (the user expects to be attacked on it and has decided not to put it in the text).
- Avoid em dashes, rule-of-three padding, and filler transitions. No sentence without a claim.

## Paper-specific framing (see also docs/paper_story_notes.md)
- The story goes from the reward-free world model to FB, not "FB improved with a world model".
- Do not ask "why not FB alone" in the intro; it is an experiment.
- Present the shared latent (world-model objective and FB objective on one encoder) as the solution, not the imagined-latent evaluation as the problem.
- Experiments section covers: main table at N = 10k (FBWM MPPI, TD-MPC upper bound, adapted world model labels-only, FB, FB on a frozen world model) and the label-budget figure. Everything else is for later.
- First drafts are clean drafts: caveats such as per-environment configuration differences are noted for later, not written yet.
