# NIE-MVI semestral project — research proposal

**Student:** Richard Černanský
**Topic category:** own topic (master's thesis work, extended with an experimental study for this course)
**Date:** 11 October 2026

---

## 1. One-paragraph summary

I have a working end-to-end 3D detection → tracking → trajectory-prediction network for
autonomous driving, **LiViP3D**, built on top of ViP3D (CVPR 2023). My version replaces
ViP3D's camera-only perception with a LiDAR branch whose BEV heatmap initialises the
transformer's object queries. On nuScenes it already beats my own reproduction of the
camera-only ViP3D 6-camera baseline on every trajectory-prediction metric
(minADE 0.96 vs 1.64, EPA 0.293 vs 0.136), while being slower and still weaker on
multi-object-tracking AMOTA. The semestral project asks two questions on top of that
working model:

1. **What does adding camera information back in actually buy?** I propose to add a
   *Spatially Modulated Co-Attention* (SMCA) camera cross-attention layer on top of the
   frozen LiDAR-only model and measure the gain per metric and per object class.
2. **How much of the performance comes from the LiDAR backbone?** I propose to swap the
   pillar-based backbone for sparse-voxel alternatives and measure accuracy vs latency.

Both are neural-architecture questions. On top of them I add the course's
nature-inspired component: the tracking and query-initialisation hyperparameters of this
model are **non-differentiable** (they sit behind Hungarian matching and hard score
gates), so gradient descent cannot reach them. I propose to tune them with population
based search (differential evolution / CMA-ES / PSO) and compare those optimisers against
random and grid search under an equal evaluation budget.

---

## 2. Background and motivation

Classical autonomous-driving stacks are modular: detect → track → predict, each stage
trained separately and communicating through thresholded, hand-tuned interfaces. ViP3D
replaced that pipeline with one differentiable network in which a set of persistent
**agent queries** carries each object through time; the trajectory predictor reads the
query features directly instead of a list of boxes. The appeal is that no information is
lost at a threshold — except that in practice the tracking gate reintroduces exactly such
a threshold, which is one of the things I want to study.

ViP3D is camera-only. Cameras give rich semantics but poor depth; LiDAR gives metric
geometry directly. My extension (LiViP3D) is the question of what the query-based
end-to-end formulation looks like when the *queries are born from LiDAR* rather than from
a fixed learned embedding:

* a LiDAR BEV feature map is produced from the point cloud (pillars → SECOND → SECONDFPN),
* a CenterPoint-style class-wise **heatmap head** predicts object centres in BEV,
* the top-K heatmap peaks become the object queries — content sampled from the BEV
  feature at the peak plus a class embedding, reference point from the peak's metric (x, y),
* a 2-layer TransFusion-style decoder refines them: layer 0 is deformable
  cross-attention over the LiDAR BEV map, layer 1 (the *proposed* part of this project)
  is SMCA cross-attention over the camera feature maps,
* surviving queries are propagated to the next frame through a memory bank + query
  interaction module (a recurrent object state), and a VectorNet/LaneGCN-style decoder
  produces 6 future trajectories per agent.

This is a natural-neural-network project in the sense the course asks for: convolutional
backbones (2D ResNet + FPN on images, sparse/pillar 3D CNN on points), transformer
self-attention and three different cross-attention inductive biases, set prediction with
Hungarian matching, and recurrent temporal state.

---

## 3. Data

**nuScenes v1.0-trainval** (nuscenes.org), the standard benchmark for this task.

| | |
|---|---|
| Scenes | 700 train / 150 val, 20 s each, keyframes at 2 Hz |
| Sensors used | 1× 32-beam LiDAR (10 accumulated sweeps, 5 features per point), 6× surround cameras at 960×544 |
| Annotations | 3D boxes + persistent instance IDs; 7 classes used (car, truck, bus, trailer, motorcycle, bicycle, pedestrian) |
| Training sample | a 3-frame clip (tracking needs temporal continuity) |
| Prediction GT | 6 s future trajectories from `nuscenes_prediction_infos_val.json` |
| Size / location | ~400 GB, read-only copy on the Perun cluster |

No new data collection is needed; the dataset is already converted to the tracking info
format this codebase consumes.

---

## 4. Starting point — what already works (preliminary results)

All numbers are on the nuScenes **val** split, all 7 classes, identical evaluation code
and identical thresholds for every row; best epoch chosen by EPA. "ViP3D 6-cam" is my own
reproduction, trained and evaluated in exactly the same harness — not a number copied
from the paper.

| Setup | minADE ↓ | minFDE ↓ | MR ↓ | EPA ↑ | AMOTA ↑ | Params | Latency |
|---|---|---|---|---|---|---|---|
| ViP3D 6-cam, camera-only (reproduced, ep 16) | 1.644 | 2.635 | 0.267 | 0.136 | 0.209 | 41.8 M | 72.7 ms |
| **LiViP3D LiDAR-only** (batch 8, ep 14) | **0.956** | **1.776** | **0.183** | **0.293** | 0.132 | 41.6 M | 130.5 ms |
| LiViP3D LiDAR-only (batch 8, ep 12) | 0.889 | 1.606 | 0.180 | 0.288 | 0.128 | — | — |
| *ViP3D as published (24 ep, 8×3090)* | *1.47* | *2.21* | *0.237* | *0.245* | *0.291* | — | — |

minADE/minFDE in metres (best of 6 modes), MR = miss rate, EPA = end-to-end prediction
accuracy, AMOTA = average multi-object tracking accuracy. The published ViP3D row uses a
different prediction-eval class subset and a longer schedule, so my reproduction is the
honest comparison.

**Reading of this:** replacing learned queries with LiDAR-heatmap queries roughly halves
the trajectory error and more than doubles EPA, at ~1.8× the inference cost. But AMOTA
*drops* relative to the camera baseline, and that is the interesting open problem: a score
analysis I ran shows true positives score 0.73 ± 0.18 and false positives 0.57 ± 0.16, so
the ranking is good but the absolute calibration is not — with that spread, a confirmed
track randomly dips below the 0.35 survival gate in ~17 % of frames and gets killed. The
detector is fine; the *hard-threshold tracking interface* is what loses the tracks. This
directly motivates the optimisation study in §5.C.

I also already have a batch-size and query-count sweep, which is the beginning of the
experimental part the course requires:

| Stage-1 variant | minADE ↓ | EPA ↑ | note |
|---|---|---|---|
| 300 queries, effective batch 8 | 0.956 | 0.293 | best |
| 300 queries, batch 16 | 0.997 | 0.277 | |
| 300 queries, batch 32 | 1.114 | 0.236 | |
| 300 queries, batch 64 | 1.061 | 0.256 | |
| 100 queries, batch 8 | 1.022 | 0.256 | |
| 100 queries, batch 16 | 1.528 | 0.234 | |

Accuracy degrades monotonically with batch size even though the learning rate was
sqrt-scaled — consistent with the usual small-batch-noise / flat-minima argument, which I
want to probe properly rather than assert.

**Training cost (measured):** stage 1 is 20 epochs ≈ 38 min/epoch ≈ 13 h on one node with
8× H200 at effective batch 8. A full val evaluation is ~100 min per checkpoint on one GPU,
and my eval harness runs 8 checkpoints in parallel on one node — which conveniently means
**one node allocation = one population of 8** for §5.C.

---

## 5. Proposed work

### A. SMCA camera fusion on top of the LiDAR-only model

The LiDAR-only model is kept frozen (backbone, neck, heatmap head) and one camera
cross-attention layer is trained on top of it for a short schedule, following
TransFusion's two-stage recipe. The layer is *Spatially Modulated Co-Attention*: each
query's 3D reference point is projected into every camera, a Gaussian centred on that
pixel with σ derived from the projected box size multiplies the attention logits, so the
query only looks where it geometrically should, and the attention is averaged over the
cameras that actually see it.

Questions to answer:

1. How much does each metric move (minADE/minFDE/MR/EPA/AMOTA), and at what latency cost?
2. Does fusion help the classes LiDAR is bad at (motorcycle, bicycle, trailer — few
   points) more than the classes it is already good at (car, pedestrian)?
3. **Does the camera help the score calibration?** This is my hypothesis for the AMOTA
   gap: appearance evidence should make the classification score less jittery across
   frames, and I can measure that directly as the per-track score standard deviation and
   the number of track fragmentations, not just as a metric delta.
4. Ablation of the modulation itself: SMCA vs plain (unmodulated) multi-head cross
   attention vs no camera layer. If the Gaussian prior is what matters, removing it should
   cost more than removing a few parameters would suggest.
5. Second ablation: letting the camera also *initialise* queries (an image-guided heatmap)
   on top of fusion, i.e. separating "fusion" from "guide".

Implementation status: the SMCA module and the two-stage schedule are already implemented
and smoke-tested in the codebase — no results yet, which is exactly what this project
produces.

### B. LiDAR backbone study

The current backbone is the cheap option: 0.2 m pillars → PointPillars scatter → SECOND →
SECONDFPN, trained from scratch. The question is how much of LiViP3D's advantage is the
*query initialisation idea* and how much is just "LiDAR features". Planned variants:

| Variant | What changes | Available |
|---|---|---|
| PointPillars 0.2 m (current) | baseline | done |
| PointPillars 0.15 m / 0.25 m | BEV resolution vs cost | config only |
| Sparse-voxel (VoxelNet-style: HardSimpleVFE + 3D sparse encoder + SECOND) | true 3D convolution instead of pillar flattening | in-tree (`SparseEncoder`) |
| Wider/deeper SECOND + SECONDFPN | capacity at fixed input encoding | config only |
| VoxelNeXt (fully sparse, no dense BEV) | stretch goal, needs external code | optional |

Each variant is judged on the same three axes: trajectory metrics, tracking metrics, and
latency/parameters — because the honest claim for this kind of work is a point on an
accuracy/cost curve, not a single best number.

### C. Nature-inspired optimisation of the non-differentiable interfaces *(course core)*

This is the part that is specific to NIE-MVI, and it is not a toy add-on — it attacks the
measured weakness from §4.

The model's inference-time behaviour is controlled by parameters no gradient ever touches,
because they live behind hard comparisons, a top-K selection and Hungarian matching:

| Parameter | Current value | Why it is arbitrary |
|---|---|---|
| `score_thresh` (confirm a new track) | 0.40 | inherited from ViP3D |
| `filter_score_thresh` (keep a track alive) | 0.35 | inherited |
| `miss_tolerance` (frames before deletion) | 5 | inherited |
| heatmap NMS kernel / per-class exemption | 3×3, exempt ped+bike | hand-picked |
| active-track suppression radius | 5×5 cells | hand-picked |
| number of queries K | 300 | swept coarsely |
| SMCA `sigma_scale`, FPN level | 1.0, level 0 | the σ heuristic is admittedly arbitrary |

Possibly even per-class thresholds (7 classes × 2 gates = 14 dimensions), since the
per-class score statistics differ a lot.

**Fitness:** EPA on a *scene-level* subset of val (subsampling must keep whole scenes, or
tracking breaks), with the winner re-evaluated on full val to check that the subset did
not overfit. EPA is non-differentiable, noisy and expensive — about 25 min per candidate
on a val quarter — which is precisely the regime population-based search is for.

**Optimisers to compare at an equal budget (~40 evaluations each, 8 in parallel per node):**
random search (the control), grid search on the two tracking gates (the honest baseline a
practitioner would do), differential evolution, CMA-ES, and particle swarm. Reported as
best-fitness-vs-evaluations curves with the variance over repeated seeds, plus the
question of whether the optimisers agree on *where* the optimum is.

**Multi-objective extension (NSGA-II):** §4 already shows a real tension — the
configuration that maximises EPA is not the one that maximises AMOTA, and adding the camera
layer costs latency. A Pareto front over (EPA, AMOTA) and over (accuracy, latency) is a
more honest summary of this model than any single scalar, and it is a textbook use of the
course's material.

**Why this is worth doing at all, not just course-compliant:** any gain found here is free
at inference time — no retraining, no extra parameters — and it tests the claim that the
"end-to-end" pipeline's remaining hand-tuned interface is costing real accuracy.

### D. Optimisation-landscape probe (small, uses existing runs)

The batch-size trend in §4 is a testable claim, not just a curve: if small batches land in
flatter minima, then the loss should rise more slowly under random parameter perturbation
for the batch-8 model than for the batch-64 one. Cheap to measure on checkpoints I already
have, and it makes the sweep an experiment rather than a table.

---

## 6. Experiment grid

The course asks for 10+ setups; this is the plan (✓ = already run, → = proposed).

| # | Setup | Axis |
|---|---|---|
| 1 | ViP3D 6-cam camera-only (reproduction) | reference |
| 2 | ✓ LiDAR-only, 300 queries, batch 8 | starting point |
| 3–6 | ✓ LiDAR-only, batch 16 / 32 / 64, and 100 queries × 2 batches | optimisation |
| 7 | → + SMCA camera layer (fusion only) | fusion |
| 8 | → + plain unmodulated cross-attention | is the Gaussian prior what matters? |
| 9 | → + SMCA + image-guided query init | fusion vs guide |
| 10 | → SMCA σ-scale / FPN-level variants | attention geometry |
| 11–13 | → backbone: pillar 0.15 m, sparse-voxel, wider SECOND | backbone |
| 14 | → best backbone + SMCA | do the two gains compose? |
| 15–19 | → random / grid / DE / CMA-ES / PSO tuning of the inference parameters | nature-inspired optimisation |
| 20 | → NSGA-II Pareto front (EPA vs AMOTA, accuracy vs latency) | multi-objective |
| 21 | → flat-minima perturbation probe on existing checkpoints | optimisation landscape |

Setups 11–14 are the compute-hungry ones (one stage-1 training each, ~13 h of one 8-GPU
node); if the cluster allocation becomes the binding constraint, the backbone list shrinks
to two variants and the search study (15–20) is preserved, because that is the
course-specific contribution.

**Protocol for every row:** same evaluation code, same class set, best epoch selected by
EPA on val, metrics reported as minADE / minFDE / MR / EPA / AMOTA / AMOTP plus parameters
and measured latency (200 timed iterations). Where a conclusion rests on a small
difference, I rerun with a different seed rather than claim it.

---

## 7. Deliverables

1. **Technical part — Jupyter notebook.** Training itself cannot live in a notebook (it is
   a multi-day, multi-GPU Slurm job on a shared cluster), so the notebook is the analysis
   and search driver, with the exact `sbatch` commands documented for reproducibility:
   training-curve and metric aggregation across all runs, the TP/FP score-distribution
   analysis, BEV/camera visualisations of the heatmap peaks and the SMCA Gaussians, the
   population-based search loop (it submits a generation of 8 candidates per node
   allocation, reads back the fitnesses, and produces the convergence and Pareto plots),
   and the final comparison tables.
2. **Documentation.** The technical report: architecture, what was changed and why, every
   experiment with its numbers, the negative results (the AMOTA gap, anything the search
   fails to improve), and the conclusions with their caveats.
3. **Presentation / blog post.** Public-facing write-up plus the 5-minute talk: one clear
   claim (LiDAR-initialised queries beat the camera-only end-to-end baseline on
   prediction), one honest limitation (tracking calibration, latency), and one result from
   the search study.

## 8. Timeline

| When | What |
|---|---|
| by 7 Nov | topic registered in the shared spreadsheet (course deadline) |
| Oct – early Nov | SMCA fusion runs + ablations (setups 7–10) |
| Nov | backbone study (11–14); start the search study once the inference parameter list is frozen |
| late Nov – mid Dec | optimiser comparison + multi-objective front (15–20), landscape probe (21) |
| Dec | notebook, documentation, blog post, rehearsal of the talk |

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cluster allocation (4 nodes, 6 h `gpu_short` jobs) becomes the bottleneck | search study uses a val subset and resumable generations; backbone list is explicitly shrinkable |
| SMCA gives no measurable gain | that is a publishable-in-a-report result, and the ablation (8) tells me whether it is the fusion or the modulation that failed; the course study does not depend on it succeeding |
| Subset fitness overfits in the search | winner always re-evaluated on full val; report both |
| Metric noise mistaken for a real effect | repeated seeds for close calls; variance reported on all search curves |
| Sparse-voxel backbone does not converge in the short schedule | compare at equal epochs *and* at equal wall-clock, and say which it is |

## 10. References

- Gu et al., *ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries*, CVPR 2023 — the baseline this work extends.
- Bai et al., *TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers*, CVPR 2022 — the two-stage recipe and SMCA fusion layer.
- Gao et al., *Fast Convergence of DETR with Spatially Modulated Co-Attention*, ICCV 2021 — SMCA.
- Yin et al., *Center-based 3D Object Detection and Tracking* (CenterPoint), CVPR 2021 — the heatmap query initialisation.
- Lang et al., *PointPillars*, CVPR 2019; Yan et al., *SECOND*, Sensors 2018 — LiDAR backbones.
- Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020.
