# Related Work & Background — Draft

## 2 Related Work

**CIL with Randomly Initialized Models.** Before foundation models became the
default backbone choice, most class-incremental learning (CIL) research
trained convolutional or residual networks (He et al. [3]) starting from
random weights, and the resulting body of work is generally grouped into
four directions. Regularization-based approaches restrict how much the
important parameters are allowed to move between tasks, either by
penalizing weight change directly (Kirkpatrick et al. [31]; Zenke et al.
[34]) or by keeping the model's predictions close to an earlier snapshot of
itself (Li & Hoiem [32]; Aljundi et al. [33]). Replay-based approaches
instead keep a small window into the past, whether that means literally
storing a subset of old exemplars (Rebuffi et al. [35]; Lopez-Paz &
Ranzato [37]; Bang et al. [40], among several others [36; 38; 39; 41; 42;
43]) or training a generator that reproduces old-class samples on demand
(Shin et al. [45]; He et al. [46]; Hu et al. [47]; Petit et al. [48]; Mocanu
et al. [44]). A fourth direction reshapes the network itself: some methods
grow the architecture with fresh capacity for every task (Aljundi et al.
[49]; Yan et al. [50]; Wang et al. [51; 52]; Zhou et al. [53]), while others
prune a shared network down into task-specific sub-networks instead (Mallya
& Lazebnik [54]; Wortsman et al. [56]; Kang et al. [57; 59]; Golkar et al.
[55]; Wang et al. [58]; Yildirim et al. [60; 61]).

**CIL with Pre-trained Models.** The picture changes considerably once a
strong pre-trained backbone, typically a Vision Transformer (Dosovitskiy et
al. [9]), is available from the start, since it already generalizes well
before any task-specific training happens at all (Ramasesh et al. [10];
Mehta et al. [11]). The open question shifts from *how do we prevent
forgetting while learning from scratch* to *how little can we touch the
model without losing that generalization* (Wu et al. [12]). One direction,
prompting, leaves the backbone fully untouched and learns a small set of
input-level tokens instead: L2P (Wang et al. [18]) was the first to select
these per instance through a learned key-query pool, DualPrompt (Wang et al.
[19]) later split the pool into task-shared and task-specific groups,
CODAPrompt (Smith et al. [20]) added attention-based reweighting to cut down
interference between prompts, and HiDePrompt (Wang et al. [21]) went a step
further and used class-wise feature statistics to imitate replay directly in
feature space. A second direction fits only a classifier on top of frozen
features. APER (Zhou et al. [22]) showed that a simple cosine-prototype
classifier, which the authors call SimpleCIL, is already a strong baseline
on its own, and RanPAC (McDonnell et al., NeurIPS 2023 [15]) pushes this
idea two steps further. First, it briefly adapts a lightweight set of
parameters on the very first task only, then freezes them for the rest of
the sequence — this first-session training stage is adopted directly from
the replay-free baseline of Panos et al. [13]. Second, once the backbone is
frozen, RanPAC lifts the extracted features into a much higher-dimensional
space through a fixed random projection, where a ridge-regularized
classifier can then be solved in closed form from accumulated Gram
statistics — no gradient descent on the classifier itself is needed at any
point. A third direction inserts adapters into the backbone's layers
directly: EASE (Zhou et al. [23]) attaches a new adapter set for every task
and concatenates all of their outputs at inference time, while MOS (Sun et
al. [24]) tries to keep that growing ensemble in check by merging adapters
together and generating pseudo-replay data for classifier alignment.
Finally, TOSCA (Yildirim et al. [R1]) takes a narrower version of the
adapter idea: instead of touching every layer, a single sparse
adapter-calibrator pair (which the authors name LuCA) is placed on the
[CLS] token right before the classifier, a fresh pair is trained per task,
and at inference the model scores each stored task module in turn and keeps
whichever prediction has the lowest entropy.

Two of the methods above are the direct starting point for this work.
RanPAC demonstrates that a closed-form classifier, fit once after a short
adaptation window, can already beat much heavier gradient-based
alternatives, but that same property is also its ceiling: nothing about the
classifier changes once the first task ends, so any structure that only
shows up in later tasks has no dedicated place to be captured. TOSCA takes
the opposite position and gives every task its own small adapter, but pays
for that flexibility at test time, since it has no choice but to run every
stored module before it can decide which prediction to trust — a cost that
keeps growing the longer the task sequence gets. [METHOD] is built to keep
the part of each method the other one lacks: TOSCA's per-task LuCA modules
supply the plasticity, and RanPAC's closed-form ridge solution supplies the
classifier, and we apply the latter at two separate levels rather than one
— a single global ridge head handles routing across all tasks, and a
second, per-task ridge head handles the actual classification once a task
has already been selected.

## 3 Background

This section lays out the CIL setting formally and then walks through the
three main families of post-training strategy that build on top of a
frozen foundation model, closing with the specific limitation in each
family that motivates the design in Section 4.

### 3.1 Class-Incremental Learning (CIL)

In CIL, a model is exposed to a sequence of $T$ disjoint tasks
$\{\mathcal{T}_1, \ldots, \mathcal{T}_T\}$, and it is trained on one task at
a time. Task $t$ comes with its own labeled dataset
$\mathcal{D}_t=\{(x_i,y_i)\}_{i=1}^{n_t}$, drawn from a set of classes
$Y_t$ that has not appeared in any earlier task, i.e. $Y_t \cap
Y_{t'}=\emptyset$ whenever $t \neq t'$. Once training on $\mathcal{D}_t$
starts, data from earlier tasks is no longer accessible, which is what
makes the setting hard: the model has to keep working on all classes seen
so far, $\mathcal{Y}=Y_1 \cup \cdots \cup Y_t$, using only what it managed
to retain internally.

Following the FM-based CIL literature (Wang et al. [18; 19]; Smith et al.
[20]; Zhou et al. [22; 23]; Sun et al. [24]), the model is written as
$f(\mathbf{x}) = W^\top \Phi(\mathbf{x})$, splitting it into a feature
extractor $\Phi: \mathcal{X} \to \mathbb{R}^d$ and a classifier
$W \in \mathbb{R}^{d \times |\mathcal{Y}|}$. For a ViT (Dosovitskiy et al.
[9]), $\Phi$ is obtained by patchifying the input into a token sequence,
prepending a learnable [CLS] token, and passing everything through
self-attention and feed-forward blocks; by convention, only the final
[CLS] token representation is kept as $\Phi(\mathbf{x})$, since it is
trained to aggregate information from the whole image. In the replay-free
setting we assume throughout (Wang et al. [18; 19]; Smith et al. [20]; Zhou
et al. [22; 23]) — the setting all of TOSCA, RanPAC, and [METHOD] operate
in — no exemplars from earlier tasks are stored, and, importantly, the task
identity of a test sample is never revealed at inference; the model has to
figure out both *what* the object is and, implicitly, *which task it came
from* at the same time.

### 3.2 Overview of Post-Training in CIL

Once the backbone is frozen, essentially every method in this space reduces
to the question of what small piece to attach on top of $\Phi(\mathbf{x})$,
and how to train it without disturbing $\Phi$ itself. We group these
strategies into three families.

**Learning Prototypical Classifiers.** The simplest strategy skips learned
parameters almost entirely: for every class $y$ seen in task $t$, a
prototype is obtained by averaging the frozen features of that class's
training examples,

$$\mathbf{p}_y = \frac{1}{n_t}\sum_{i=1}^{n_t} \Phi(\mathbf{x}_i), \tag{1}$$

and a test sample is classified by whichever prototype it lies closest to.
RanPAC [15] keeps this basic recipe but replaces the distance computation
with something considerably stronger: features are first passed through a
fixed random projection with a nonlinearity, $\phi(\mathbf{x}) =
\mathrm{ReLU}(\Phi(\mathbf{x})^\top P)$ with $P \in \mathbb{R}^{d \times M}$,
$M \gg d$, and the classifier weights are then obtained in closed form as
$W^\star = (G + \lambda I)^{-1} C$, where $G$ and $C$ are Gram and
class-sum matrices accumulated over $\phi(\mathbf{x})$ rather than solved
by gradient descent at all. Together with the first-session training stage
described above, this lets RanPAC skip training a classifier from scratch
for every incoming task — it only ever updates $G$ and $C$. The trade-off
is that once the first task's adaptation is frozen, the feature space
itself no longer moves, so the only thing that can still adjust to new
classes is the accumulation of statistics on top of it; there is nowhere
for genuinely task-specific structure, arriving after task one, to be
represented.

**Learning Prompts.** Rather than modifying the classifier, this family
modifies what the backbone sees as input. A pool of learnable prompt
vectors $\mathcal{P}=\{P_1,\dots,P_M\}$, each paired with a trainable key,
is inserted either into the embedding layer alone or across several
transformer blocks, and an instance is routed to the prompts whose keys
are closest to its own query embedding under the *frozen* backbone (Wang et
al. [18; 19]; Smith et al. [20]; Wang et al. [21]). Because prompts are
shared parameters rather than one-per-task modules, this approach keeps the
overall parameter count small and the backbone completely untouched, which
is a real advantage for stability. The difficulty shows up at retrieval
time: as more tasks accumulate, prompts belonging to different tasks can
end up describing similar-looking classes, and since the key space used
for routing is fixed from the start, it has no way to later separate
prompts that turn out to conflict — which is exactly the kind of
interference that leads to forgetting in this family.

**Learning Adapters.** The third family edits the backbone directly by
inserting small bottleneck modules into its layers. For an intermediate
representation $\mathbf{z}$, a task-specific adapter set $\mathcal{A}_t =
\{A_1, \ldots, A_N\}$ (one module per transformer layer, in the layer-wise
case) applies

$$A(\mathbf{z}) = \mathbf{z} + \psi(\mathbf{z}W_{down})W_{up}, \qquad
W_{down}\in\mathbb{R}^{d\times r},\ W_{up}\in\mathbb{R}^{r\times d}, \tag{4}$$

with $\psi$ a nonlinearity and $r \ll d$ (Zhou et al. [22; 23]; Sun et al.
[24]). Placing one such module in every layer, for every task, is
effective but expensive: the parameter count grows as $\mathcal{O}(T\!\cdot\!
N\!\cdot\! dr)$, and small representational shifts introduced at every
layer tend to compound by the time they reach the classifier. TOSCA [R1]
avoids the layer-wise placement entirely and instead applies a single
adapter-calibrator pair only to the final [CLS] representation,

$$\Phi(\mathbf{x})' = C(A(\Phi(\mathbf{x}))),$$

which keeps the parameter count fixed at $\mathcal{O}(dr)$ no matter how
deep the backbone is, and trains it with an added $\ell_1$ penalty that
pushes different tasks' modules toward disjoint, sparse sets of active
weights. The cost shows up at inference rather than training: with no task
label available, TOSCA has to run every stored module on a given input and
keep the prediction with the lowest entropy, so scoring a single test
sample takes work proportional to the number of tasks seen so far.

Put next to each other, these three families trade the same two things
against each other in different places. Prototypical classifiers, sharpened
by RanPAC's random projection and closed-form solve, are cheap and stable
but stop adapting after the first task; prompts and adapters both restore
some form of ongoing adaptation, at the cost of either retrieval conflicts
or, in TOSCA's case specifically, inference cost that scales with $T$.
[METHOD] is an attempt to take the piece of RanPAC that already works well
— a routing-and-classification mechanism that needs no gradient descent
after the features are fixed — and pair it with TOSCA's per-task LuCA
modules, so that plasticity is no longer something that has to be paid for
with a linear scan over every task at test time. Section 4 gives the full
formulation.

---

## References used above

Citation numbers reused from TOSCA's own bibliography (arXiv:2502.14762v2,
pp. 13–16) wherever the same source applies; [R1] and [R2] are added for
sources TOSCA cannot cite about itself and about RanPAC, respectively, since
those needed direct verification rather than trusting TOSCA's paraphrase.

- **[3]** He, Zhang, Ren, Sun. Deep residual learning for image
  recognition. CVPR, 2016.
- **[9]** Dosovitskiy, Beyer, Kolesnikov, et al. An image is worth 16x16
  words: Transformers for image recognition at scale. ICLR, 2021.
- **[10]** Ramasesh, Lewkowycz, Dyer. Effect of scale on catastrophic
  forgetting in neural networks. ICLR, 2022.
- **[11]** Mehta, Patil, Chandar, Strubell. An empirical investigation of
  the role of pre-training in lifelong learning. JMLR, 24(214), 2023.
- **[12]** Wu, Swaminathan, Li, Ravichandran, Vasconcelos, Bhotika, Soatto.
  Class-incremental learning with strong pre-trained models. CVPR, 2022.
- **[13]** Panos, Kobe, Olmeda Reino, Aljundi, Turner. First session
  adaptation: A strong replay-free baseline for class-incremental
  learning. ICCV, 2023.
- **[15] / [R2]** McDonnell, Gong, Parvaneh, Abbasnejad, van den Hengel.
  RanPAC: Random Projections and Pre-trained Models for Continual
  Learning. **NeurIPS 2023** (arXiv:2307.02251). TOSCA's own bibliography
  misdates this as "NeurIPS, 2024" — confirmed from RanPAC's own title
  page ("37th Conference on Neural Information Processing Systems (NeurIPS
  2023)"); use 2023.
- **[17]** Janson, Zhang, Aljundi, Elhoseiny. A simple baseline that
  questions the use of pretrained-models in continual learning. NeurIPS
  Workshop on Distribution Shifts, 2022.
- **[18]** Wang, Zhang, Lee, et al. Learning to prompt for continual
  learning. CVPR, 2022. (L2P)
- **[19]** Wang, Zhang, Ebrahimi, et al. DualPrompt: Complementary
  prompting for rehearsal-free continual learning. ECCV, 2022.
- **[20]** Smith, Karlinsky, Gutta, et al. CODA-Prompt: Continual
  decomposed attention-based prompting for rehearsal-free continual
  learning. CVPR, 2023.
- **[21]** Wang, Xie, Zhang, Huang, Su, Zhu. Hierarchical decomposition of
  prompt-based continual learning: Rethinking obscured sub-optimality.
  NeurIPS, 2023. (HiDePrompt)
- **[22]** Zhou, Cai, Ye, Zhan, Liu. Revisiting class-incremental learning
  with pre-trained models: Generalizability and adaptivity are all you
  need. IJCV, 2024. (APER / SimpleCIL)
- **[23]** Zhou, Sun, Ye, Zhan. Expandable subspace ensemble for
  pre-trained model-based class-incremental learning. CVPR, 2024. (EASE)
- **[24]** Sun, Zhou, Zhao, Gan, Zhan, Ye. MOS: Model surgery for
  pre-trained model-based class-incremental learning. **AAAI, 2025** (not
  2024 as TOSCA's own text implies — the arXiv preprint appeared in late
  2024, but the paper itself is AAAI 2025; verified independently, see
  `references.bib`).
- **[31]–[34]** Kirkpatrick et al. (EWC), PNAS 2017; Li & Hoiem (LwF),
  TPAMI **2018** (print/issue year; the DOI's early-access date of 2017 is
  what's commonly, and incorrectly, cited), Aljundi et al. (MAS), ECCV
  2018; Zenke et al. (SI), ICML 2017.
- **[35]–[43]** Rebuffi et al. (iCaRL), CVPR 2017; Prabhu et al. (GDumb),
  ECCV 2020; Lopez-Paz & Ranzato (GEM), NeurIPS 2017; Wu et al., CVPR
  2019; Zhao et al., CVPR 2020; Bang et al. (Rainbow Memory), CVPR 2021;
  Liu & Sun (RMM), NeurIPS 2021; Arani et al., ICLR 2022; Sarfraz et al.,
  ICLR 2023.
- **[44]–[48]** Mocanu et al., arXiv 2016; Shin et al., NeurIPS 2017; He
  et al., BMVC 2018; Hu et al., ICLR 2019; Petit et al. (FeTrIL), WACV
  2023.
- **[49]–[53]** Aljundi et al. (Expert Gate), CVPR 2017; Yan et al. (DER),
  CVPR 2021; Wang et al. (FOSTER), ECCV 2022; Wang et al., CVPR 2023;
  Zhou et al. (MEMO), ICLR 2023.
- **[54]–[61]** Mallya & Lazebnik (PackNet), CVPR 2018; Golkar et al.,
  NeurIPS 2019; Wortsman et al. (Supermasks in Superposition), NeurIPS
  2020; Kang et al., ICML 2022; Wang et al. (SparCL), NeurIPS 2022; Kang
  et al., ICLR 2023; Yildirim et al., CPAL 2024; Yildirim et al., arXiv
  2025.
- **[R1]** Yildirim, Gok Yildirim, Vanschoren. Unlocking [CLS] Features
  for Continual Post-Training. TMLR, 2026. arXiv:2502.14762. (This is the
  paper being extended — cannot appear in its own numbering.)

Verification note: the RanPAC-specific claims above (the Eq. (5)-equivalent
ridge form, and the attribution of first-session training to Panos et al.
[13]) were checked directly against RanPAC's own PDF (arXiv:2307.02251v3),
not just against TOSCA's summary of it — RanPAC's Section 4.4 states
explicitly, twice, that it combines its random-projection layer with a PETL
method trained using "first-session" adaptation "as carried out by [65,
37]," where RanPAC's own reference [37] is Panos et al., ICCV 2023. All
other citation numbers were checked against TOSCA's reference list directly
(pp. 13–16) at the exact claim each is attached to.
