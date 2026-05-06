# Potencial-usage-of-Riemannian-Geometrics-in-Phonetic-Analysis
========================================================================
 Geometric Modeling of Hesitation in Second-Language (L2) Speech
 A Riemannian / SRVF-based Research Pipeline
========================================================================

Overview
--------
This pipeline models speech hesitation as distortion in the geometry of
speech trajectories. Each utterance is represented as a multi-dimensional
time-series γ(t) ∈ ℝ^d encoding prosodic and temporal features (energy,
pausing, pitch, speech rate). Trajectories are lifted into the Square-Root
Velocity Function (SRVF) representation, which lives on the unit Hilbert
sphere in L²([0,1], ℝ^d). Distances in this space are invariant to
re-parameterisation (elastic distances), making them robust to speaking rate
variation — a critical confound in L1/L2 comparisons.

Theoretical Motivation
-----------------------
• SRVF representation: Fisher–Rao metric on the space of curves is
  computationally intractable. Srivastava et al. (2011) showed that the
  square-root velocity transform converts elastic distances into ordinary
  L² distances on the unit sphere, making computation straightforward.
• Riemannian geometry: The space of SRVFs with the geodesic distance is
  a Riemannian manifold. The "mean" trajectory (Fréchet mean) on this
  manifold is the intrinsic centre of a speaker group, not a pointwise
  average — it is geometrically correct.
• Hesitation as trajectory distortion: Native speech follows tight,
  low-variance trajectories in SRVF space. Hesitation, filled pauses, and
  dysfluency perturb the energy/pause profile, increasing geodesic distance
  from the native centroid. This deviation is our hesitation score.

References
----------
Srivastava, A., Klassen, E., Joshi, S. H., & Jermyn, I. H. (2011).
  Shape analysis of elastic curves in Euclidean spaces. IEEE TPAMI.

Tucker, J. D., Wu, W., & Srivastava, A. (2013).
  Generative models for functional data using phase and amplitude separation.
  Computational Statistics & Data Analysis.

Characteristics relevant to L2 research are discussed in:
  Skehan, P. (1998). A Cognitive Approach to Language Learning. OUP.
  Cucchiarini, C., Strik, H., & Boves, L. (2002). Quantitative assessment
    of second language learners' fluency. JASA.

Author:  SHuojia Wang
License: MIT
