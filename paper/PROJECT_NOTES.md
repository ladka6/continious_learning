# Project Notes — [METHOD] paper

Running reference for this paper folder. Update as the paper evolves; this is not
part of the submission, just a working memory of decisions and state.

## What this paper is

[METHOD] (placeholder name — title page currently says "Adini Gate Koydum",
author Elif Ceren Gok) is a class-incremental learning (CIL) method described as
"a child of TOSCA and RanPAC":

- From **RanPAC**: first-session adaptation (lightweight PETL/adapter params
  trained only on task 0, then frozen) + closed-form ridge classification
  (`W* = (G + λI)^-1 C`, no gradient descent on the classifier).
- From **TOSCA**: a fresh sparse LuCA adapter-calibrator module
  (`L(z) = C(A(z))`) trained per task on the ViT's final `[CLS]` token.
- The actual novelty: **two-level ridge routing**, replacing both TOSCA's
  cosine-prototype classifier and its O(T) entropy-scan inference cost.
  - A **global ridge head** on frozen, un-adapted features does top-1 task
    routing.
  - A **per-task ridge head** on that task's LuCA-adapted features does
    within-task classification once routed.
  - This is why the paper's own results row keeps RanPAC-level inference cost
    (~35.6 GFLOPs, same as RanPAC) while beating TOSCA on accuracy — it never
    needs the linear scan over all stored task modules that TOSCA does.

Source codebases (read-only, for reference, not part of this repo):
- `~/Projects/tue/tosca` — original TOSCA implementation.
- `~/Projects/tue/continue_learning` — this method's implementation
  (`models/tosca.py` has the two-level ridge routing logic, see
  `_get_global_routed_ridge_logits`).

## Files in this folder

- `paper.tex` — the paper. Currently has Introduction (stub), Related Work,
  Background, Experiments (with real result tables), and forward-references a
  not-yet-written `sec:method` (Methodology) — that section still needs to be
  written before the paper compiles without warnings.
- `references.bib` — 64 entries, independently verified (see below). Key
  citation: TOSCA itself is `yildirim2026unlocking`.
- `related_work_background.md` — earlier markdown draft of Sections 2–3,
  superseded by `paper.tex` but kept as the verification/reasoning trail
  (has a "References used above" section with notes on why each citation
  was chosen).

## Citation verification methodology (important — don't skip if adding more refs)

This paper explicitly does **not** trust TOSCA's own bibliography as ground
truth, even though TOSCA was the initial pointer to related work. All 63
non-self references were independently re-verified against primary sources
(arXiv, DBLP, conference proceedings, Semantic Scholar) via parallel research
agents. This caught real errors in TOSCA's own PDF/bib, including:

- Wrong years from early-access vs. print confusion (e.g. `masana2023...`,
  `delange2022continual`, `li2018learning` — TPAMI early-access dates ≠ print
  dates).
- Wrong venue tier (`sun2025mos` is AAAI 2025, not the arXiv-preprint year
  TOSCA implied; `golkar2019continual` is a NeurIPS 2019 *workshop* paper, not
  main track).
- **Dropped/garbled co-authors** — found the same bug pattern twice
  independently: `liu2021rmm` (TOSCA's PDF dropped Bernt Schiele, printed
  "Qianru Sun, and Qianru Sun") and `hu2019overcoming` (garbled duplicate
  surname hiding a different real author list). Worth treating as a
  systematic OCR issue in TOSCA's own pipeline, not a one-off — re-check
  author lists carefully if pulling more citations from TOSCA's bib.

If adding new references later: verify against the primary source directly
(not against TOSCA's bib, not against Semantic Scholar's auto-scraped fields
alone — cross-check with the actual venue page or arXiv abstract page).

## Writing conventions for Sections 2–3

- Structural template borrowed from TOSCA's paper (arXiv:2502.14762,
  "Unlocking [CLS] Features for Continual Post-Training") — same paragraph
  skeleton, but prose rewritten from scratch. Do not copy TOSCA's sentences
  directly; user flagged this explicitly once already.
- Style: `ege-writing-voice` skill applied at low-moderate density (a few
  structural bridging sentences, not heavy ESL-slip injection) — appropriate
  for submission-grade writing rather than a thesis/blog-register document.
- **Don't forward-reference experimental numbers before the method they
  belong to has been formally introduced.** Caught and reverted once already:
  had added accuracy/efficiency callouts (citing `Table~\ref{tab:main-results}`
  etc.) into the Related Work and Background closing paragraphs, before
  `sec:method` exists. Both paragraphs should end on a plain
  `Section~\ref{sec:method} gives the full formulation.` pointer instead —
  save the numbers for when the method has actually been defined.

## Known open items

- `sec:method` (Methodology) not written yet — needed for the paper to
  compile without "undefined reference" warnings and for the numeric
  comparisons to have somewhere to land.
- Table 1/2 in Experiments label the paper's own method's result row
  `TOSCA (ours)` — almost certainly a leftover from before `[METHOD]` was
  turned into a placeholder. Not yet fixed; flag to user before touching,
  since it's their table.
- `references.bib` has 18 entries not currently cited anywhere in
  `paper.tex` (mostly neuroscience/general-CIL-survey citations from TOSCA's
  original bib that this paper's rewritten prose didn't end up needing):
  `bhandari2025task, biederman1987recognitionbycomponents,
  biederman1993recognizing, dascoli2021convit, delange2022continual,
  friston2009cortical, grossberg2012studies, janson2022simple,
  liu2021swin, masana2023classincremental, mccloskey1989catastrophic,
  ning2024sparse, radford2021learning, wang2022sprompts,
  wang2023orthogonal, zhang2022reshaping, zhang2023slca,
  zhou2024classincremental`. Leave them in for now — Methodology/Discussion
  may still need some of them (`wang2023orthogonal`, `zhang2023slca` look
  like plausible Methodology citations).
- LaTeX build note: `newtxtext`/`newtxmath` packages aren't installed in
  this sandbox's TeX Live, so local `pdflatex` test-compiles need those two
  lines commented out to verify structure/citations. Not a real bug in the
  paper — just missing on this machine.

## Quick citation sanity check (rerun after editing paper.tex or references.bib)

```bash
cd ~/Projects/tue/continue_learning/paper
grep -oE '\\cite[pt]?\{[^}]+\}' paper.tex | sed -E 's/\\cite[pt]?\{//; s/\}//' | tr ',' '\n' | sort -u > /tmp/used_keys.txt
grep -oE '^@[a-zA-Z]+\{[^,]+,' references.bib | sed -E 's/^@[a-zA-Z]+\{//; s/,$//' | sort -u > /tmp/bib_keys.txt
echo "=== used but not defined (should be empty) ==="
comm -23 /tmp/used_keys.txt /tmp/bib_keys.txt
rm /tmp/used_keys.txt /tmp/bib_keys.txt
```
