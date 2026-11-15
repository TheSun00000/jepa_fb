# Writing checklist: text that does not read as machine-written

Use this on every report, journal entry, tracker note, and paper section before handing it over. Fix problems by
rewriting the sentence with a concrete fact, not by swapping in a synonym. One flagged word alone does not matter. A
cluster of them does, and so does a whole paragraph with no specific content.

Sources are listed at the end. The word lists change over time (models learn to avoid the best-known tics), so the
structural checks in sections 2-4 matter more than the word list.

## 0. The one-minute pass

1. Does each paragraph contain a number, a run folder, a named method, or an equation? If not, cut it or add one.
2. Search the text for the words in section 1. Rewrite each hit unless it is the plainest choice available.
3. Search for `—` (em dash), `not only`, `not just`, `rather than`, `In summary`, `Overall`. Rewrite most hits.
4. Count the lists of exactly three. Keep a triplet only if there really are three items.
5. Read the first and last sentence of each paragraph. Delete any that only announce or restate.
6. Check that every claim of importance ("crucial", "key", "significant") is backed by a result, or remove it.

## 1. Words and phrases to avoid

These come from the Kobak et al. study of PubMed abstracts (words whose frequency jumped after 2023) and from
Wikipedia's field guide. Plainer replacements are given where one exists.

**Verbs**
- delve (into) -> study, look at, examine
- underscore, highlight (as filler) -> show
- showcase -> show
- leverage, harness, utilize -> use
- foster, cultivate -> help, build, encourage (or cut)
- unlock, unleash, empower, elevate, streamline, bolster, spearhead, navigate (figurative), embark
- shed light on -> show, explain
- pave the way for, set the stage for -> lead to, allow
- resonate with, align with (figurative)
- garner, boast (for "has")

**Adjectives and nouns**
- crucial, pivotal, vital, paramount, key (as filler) -> important, or say why it matters
- comprehensive, holistic, robust (when not about statistical robustness), seamless, cutting-edge, groundbreaking,
  innovative, novel (when every paper says novel), transformative, unprecedented, remarkable, notable
- intricate, multifaceted, nuanced, complex interplay
- tapestry, landscape, realm, testament, cornerstone, journey, symphony, beacon
- insights, valuable insights -> results, findings (or say what was found)
- evolving landscape, rapidly evolving field, ever-changing world

**Adverbs and connectors**
- notably, particularly, additionally, moreover, furthermore, importantly, crucially, ultimately, essentially
- meticulously, seamlessly, effectively (filler)
- It is worth noting that, It is important to note that, Interestingly, -> state the fact
- That being said, With that in mind, In light of this, In today's ...
- In summary, In conclusion, Overall, To sum up (at the end of a paragraph or short section)

**Inflated copulas**: "serves as", "stands as", "acts as", "represents", "features", "offers", "boasts" where "is" or
"has" works. Write "B is a linear map", not "B serves as a linear map".

**Vague attribution**: "experts argue", "studies show", "it is widely believed", "researchers have noted", "industry
reports". In a paper: cite the paper or delete the clause.

**Chatbot leftovers** (never in a document): "Certainly!", "Great question", "I hope this helps", "Let me know if",
"Here is a detailed overview", "As of my last update", "based on available information", "Would you like me to".

## 2. Sentence patterns to avoid

- **Negative parallelism**: "not only X but also Y", "it's not X, it's Y", "X rather than Y", "no X, no Y, just Z".
  Say what it is.
- **Rule of three**: "fast, scalable, and robust", triplets of adjectives or short phrases used for rhythm. Use the
  number of items you actually have; two is fine, one is often better.
- **Trailing -ing clause that adds fake analysis**: "..., highlighting the importance of planning", "..., ensuring
  stable training", "..., reflecting the method's flexibility". Either the claim is shown by a result (then cite it)
  or it goes.
- **Significance padding**: "plays a crucial role in", "is a testament to", "marks a pivotal step", "has significant
  implications for", "opens new avenues". Replace with the specific consequence, or cut.
- **False range**: "from X to Y" where X and Y are not ends of a scale ("from locomotion to navigation").
- **Hedge stacking**: "may potentially help to some extent". One hedge per claim, placed where the uncertainty is,
  and quantified when possible (seeds, std, CI).
- **Announcing**: "In this section, we will explore...", "Let's take a closer look". Start with the content.
- **Echo summary**: a closing sentence that repeats the paragraph. Cut it.
- **Elegant variation**: calling the same thing "the model", "our framework", "the proposed approach", "the system" in
  turn. Pick one name and keep it (e.g. always "sepB").
- **Uniform rhythm**: every sentence 15-25 words, every paragraph four sentences. Vary length; a short sentence
  after a long one is fine.
- **Colon reveal**: "The result: a model that..." / "The key insight? ...". Write a normal sentence.

## 3. Formatting habits to avoid (in prose documents and the paper)

- Em dashes (—) as the default punctuation. Use a comma, a colon, parentheses, or a new sentence.
- Bold scattered through running text, or bold lead-ins on every bullet.
- Bullet lists where the content is an argument (reasoning needs sentences and connectives).
- Title Case Headings. Use sentence case unless the venue requires otherwise.
- Emoji in headings or lists. Horizontal rules (`---`) between every section.
- Headings like "Challenges and Future Directions", "Key Takeaways", "Conclusion" on a three-paragraph note.
- Curly quotes pasted into LaTeX or Markdown sources; use the source's own convention.

## 4. Content checks (the ones that matter most)

- **Specific over general.** "MPPI improves return by 12% on walker (5 seeds, runs X, Y)" beats "planning
  substantially improves performance".
- **Claim size matches evidence.** No "consistently", "significantly", "across all tasks" unless the table says so.
  "Significant" only with a test.
- **Say what failed.** Negative and mixed results stated plainly, not hidden behind "further work is needed".
- **No made-up references or numbers.** Every citation checked to exist; every number traceable to a run folder
  (project rule: no result without its run folders).
- **No puffery about prior work or our own.** Describe what a method does, not how impressive it is.
- **Limitations are concrete.** "Only three dm_control domains and one antmaze map", not "our approach has some
  limitations".
- **Write for the reader who knows the field.** Do not define RL, MPC, or successor features at length in a paper's
  method section; do define our own variants.

## 5. Before and after

(The numbers below are made up for the example.)

> Notably, our comprehensive framework not only leverages the forward-backward representation but also harnesses a
> latent world model, highlighting the crucial role of planning in zero-shot RL.

> We plan with MPPI in the latent space of the world model, using F as the terminal value. On walker this adds
> 12 points of return over the FB policy (table 2).

## Sources

- Wikipedia, WikiProject AI Cleanup, "Signs of AI writing":
  https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Kobak, González-Márquez, Horvát, Lause, "Delving into ChatGPT usage in academic writing through excess vocabulary"
  (arXiv 2406.07016): https://arxiv.org/abs/2406.07016
- GPTZero, most common AI vocabulary: https://gptzero.me/news/most-common-ai-vocabulary/
- Grammarly, common AI words: https://www.grammarly.com/blog/ai-writing/common-ai-words/
- TechCrunch on the Wikipedia guide (2025-11-20):
  https://techcrunch.com/2025/11/20/the-best-guide-to-spotting-ai-writing-comes-from-wikipedia/
