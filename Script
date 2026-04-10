"""
================================================================================
 Geometric Modeling of Hesitation in Second-Language (L2) Speech
 A Riemannian / SRVF-based Research Pipeline
================================================================================

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

Author:  Research Prototype (adapt freely)
License: MIT
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
import itertools
from pathlib import Path

# ── Numerical / scientific ─────────────────────────────────────────────────────
import numpy as np
from scipy import stats, interpolate, signal
from scipy.spatial.distance import squareform
from scipy.linalg import svd

# ── Audio ──────────────────────────────────────────────────────────────────────
import librosa
import librosa.effects

# ── Machine learning / dimensionality reduction ────────────────────────────────
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Visualisation ──────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Georgia"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
})

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    SINGLE_SAMPLE_MODE = True   # ← add this
    """
    Central configuration object. Adjust paths and hyper-parameters here
    before running the pipeline.
    """
    # ── Paths ──────────────────────────────────────────────────────────────────
    L1_DIR = Path("C:/Users/13527/Desktop/Barbosa/RimannianGeometrics/pretest/nativespeaker")
    L2_DIR = Path("C:/Users/13527/Desktop/Barbosa/RimannianGeometrics/pretest/pretest.wav")          # directory of L2-learner .wav
    OUTPUT_DIR    = Path("output_figures")   # where plots are saved

    # ── Audio preprocessing ────────────────────────────────────────────────────
    SR            = 16_000      # target sample rate (Hz)
    HOP_LENGTH    = 160         # 10 ms hop (at 16 kHz)
    N_FFT         = 512
    FRAME_SECONDS = 0.010       # seconds per frame

    # ── Pause detection ────────────────────────────────────────────────────────
    # Method: 'energy' | 'webrtcvad_proxy' | 'zcr_energy'
    PAUSE_METHOD  = "zcr_energy"
    ENERGY_PERCENTILE_THRESHOLD = 25   # bottom 25 % of energy → pause
    MIN_PAUSE_FRAMES  = 3              # ≥ 30 ms to count as a pause
    MIN_SPEECH_FRAMES = 5              # hysteresis: speech must persist ≥ 50 ms

    # ── Feature dimensions to include ─────────────────────────────────────────
    USE_PITCH     = True
    USE_RATE      = True                # approximate syllable/speech rate
    TRAJECTORY_DIM = None              # auto-computed from flags above

    # ── Resampling ─────────────────────────────────────────────────────────────
    TRAJ_LENGTH   = 200                # all trajectories resampled to this length

    # ── Pitch estimation ──────────────────────────────────────────────────────
    F0_FMIN       = 75.0               # Hz – lowest expected F0
    F0_FMAX       = 400.0              # Hz – highest expected F0

    # ── Statistical testing ───────────────────────────────────────────────────
    ALPHA         = 0.05

    # ── Plot colours ──────────────────────────────────────────────────────────
    COLOR_L1      = "#2166AC"
    COLOR_L2      = "#D6604D"

    @classmethod
    def feature_dim(cls) -> int:
        d = 2  # energy + pause always included
        if cls.USE_PITCH: d += 1
        if cls.USE_RATE:  d += 1
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(l1_dir: Path, l2_dir: Path, sr: int = Config.SR):
    """
    Load all .wav files from L1 and L2 directories.

    Returns
    -------
    audio_l1 : list of (label, waveform_np) for native speakers
    audio_l2 : list of (label, waveform_np) for L2 learners
    """
    def _load_dir(directory: Path, group_tag: str):
        records = []
        wav_files = sorted(directory.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(
                f"No .wav files found in {directory}. "
                "Please populate data/l1 and data/l2 with .wav files."
            )
        for p in wav_files:
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            records.append({"label": f"{group_tag}_{p.stem}", "group": group_tag,
                             "waveform": y, "path": str(p)})
            print(f"  Loaded [{group_tag}] {p.name}  ({len(y)/sr:.2f}s)")
        return records

    print("\n── Loading dataset ──────────────────────────────────────────────")
    l1 = _load_dir(l1_dir, "L1")
    l2 = _load_dir(l2_dir, "L2")
    print(f"  {len(l1)} L1 files, {len(l2)} L2 files loaded.")
    return l1, l2


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Energy (RMS) ──────────────────────────────────────────────────────────

def extract_energy(y: np.ndarray, hop: int = Config.HOP_LENGTH,
                   n_fft: int = Config.N_FFT) -> np.ndarray:
    """
    Frame-level RMS energy. Returns a 1-D array of shape (T,).

    We use the log-compressed RMS (in dB-like units) because the dynamic
    range of speech energy spans several orders of magnitude; log compression
    yields a roughly Gaussian distribution amenable to threshold-based
    methods.
    """
    rms = librosa.feature.rms(y=y, frame_length=n_fft,
                               hop_length=hop)[0]          # shape (T,)
    rms = np.maximum(rms, 1e-10)                            # guard log(0)
    log_rms = 20.0 * np.log10(rms)                         # dB scale
    return log_rms


# ── 3b. Pause detection ───────────────────────────────────────────────────────

def detect_pauses_zcr_energy(y: np.ndarray,
                              sr: int = Config.SR,
                              hop: int = Config.HOP_LENGTH,
                              n_fft: int = Config.N_FFT,
                              energy_pct: float = Config.ENERGY_PERCENTILE_THRESHOLD,
                              min_pause: int = Config.MIN_PAUSE_FRAMES,
                              min_speech: int = Config.MIN_SPEECH_FRAMES
                              ) -> np.ndarray:
    """
    Improved pause detection using joint energy-ZCR criterion.

    A frame is labelled as a PAUSE when:
        energy < energy_threshold   AND   zcr > zcr_threshold
    (silence has low energy AND high zero-crossing rate from background noise,
    whereas fricatives have high ZCR but also high energy).

    Additionally, a hysteresis filter removes spurious short on/off switches.

    Returns
    -------
    pause_signal : np.ndarray, shape (T,), values in {0.0, 1.0}
    """
    rms    = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    zcr    = librosa.feature.zero_crossing_rate(y, frame_length=n_fft,
                                                hop_length=hop)[0]
    T      = min(len(rms), len(zcr))
    rms, zcr = rms[:T], zcr[:T]

    e_thresh   = np.percentile(rms, energy_pct)
    zcr_thresh = np.percentile(zcr, 50)          # median ZCR

    is_silent  = (rms < e_thresh) & (zcr > zcr_thresh)

    # ── Hysteresis smoothing ───────────────────────────────────────────────────
    pause_signal = is_silent.astype(float)
    # Suppress very short pauses
    in_pause   = False
    run        = 0
    smoothed   = pause_signal.copy()
    for i in range(T):
        if smoothed[i] == 1.0:
            run += 1
        else:
            if in_pause and run < min_pause:
                smoothed[i - run: i] = 0.0
            run = 0
            in_pause = False
        if smoothed[i] == 1.0 and run == 1:
            in_pause = True

    return smoothed


def detect_pauses_energy(y: np.ndarray,
                         hop: int = Config.HOP_LENGTH,
                         n_fft: int = Config.N_FFT,
                         energy_pct: float = Config.ENERGY_PERCENTILE_THRESHOLD,
                         min_pause: int = Config.MIN_PAUSE_FRAMES) -> np.ndarray:
    """
    Fallback: percentile-threshold energy-only pause detector.
    """
    rms     = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    thresh  = np.percentile(rms, energy_pct)
    silent  = (rms < thresh).astype(float)

    # Suppress bursts shorter than min_pause
    kernel  = np.ones(min_pause) / min_pause
    smooth  = np.convolve(silent, kernel, mode="same")
    return (smooth > 0.5).astype(float)


def get_pause_signal(y, sr=Config.SR, hop=Config.HOP_LENGTH,
                     method=Config.PAUSE_METHOD) -> np.ndarray:
    if method == "zcr_energy":
        return detect_pauses_zcr_energy(y, sr=sr, hop=hop)
    else:
        return detect_pauses_energy(y, hop=hop)


# ── 3c. Pitch (F0) ────────────────────────────────────────────────────────────

def extract_pitch(y: np.ndarray, sr: int = Config.SR,
                  hop: int = Config.HOP_LENGTH,
                  fmin: float = Config.F0_FMIN,
                  fmax: float = Config.F0_FMAX) -> np.ndarray:
    """
    Extract fundamental frequency (F0) using librosa's YIN algorithm.

    Unvoiced frames are set to 0 Hz. The resulting contour is mean-subtracted
    and scaled by the per-utterance standard deviation so that speaker-
    specific F0 range is factored out; only the *shape* of the F0 contour
    (rises/falls) is retained, which correlates with hesitation marking
    (e.g., prolonged level tone on a filled pause).
    """
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
    # Voiced/unvoiced mask: YIN returns fmax for unvoiced frames
    voiced = f0 < fmax * 0.98
    f0[~voiced] = 0.0
    # Normalise by voiced-frame statistics to remove speaker range
    voiced_vals = f0[voiced]
    if voiced_vals.size > 1:
        mu, sigma = voiced_vals.mean(), voiced_vals.std() + 1e-8
        f0[voiced] = (f0[voiced] - mu) / sigma
    return f0


# ── 3d. Approximate speech rate (syllable nucleus density) ───────────────────

def extract_speech_rate(y: np.ndarray, sr: int = Config.SR,
                        hop: int = Config.HOP_LENGTH,
                        n_fft: int = Config.N_FFT,
                        window_frames: int = 50) -> np.ndarray:
    """
    Estimate local speech rate as the density of vowel-nucleus candidates
    within a sliding window.

    Vowel nuclei are approximated as local energy peaks in the 300–3000 Hz
    band (the first two formant region), following a simplified version of
    the de Jong & Wempe (2009) Praat script approach.

    This gives a frame-level rate curve that rises during fluent speech and
    drops during pauses or syllable elongation — a known hesitation cue.

    Returns
    -------
    rate_curve : np.ndarray, shape (T,)
    """
    # Band-pass filter to formant region
    sos = signal.butter(4, [300, 3000], btype="bandpass",
                        fs=sr, output="sos")
    y_bp = signal.sosfilt(sos, y)

    rms  = librosa.feature.rms(y=y_bp, frame_length=n_fft, hop_length=hop)[0]
    T    = len(rms)

    # Find local peaks in band-passed energy → syllable nuclei candidates
    peaks, _  = signal.find_peaks(rms, distance=int(0.05 * sr / hop),
                                  prominence=rms.std() * 0.3)
    nucleus   = np.zeros(T)
    nucleus[peaks] = 1.0

    # Convolve with rectangular window to get local nucleus density
    win          = np.ones(window_frames) / window_frames
    rate_curve   = np.convolve(nucleus, win, mode="same")
    return rate_curve


# ── 3e. Assemble multi-dimensional trajectory ─────────────────────────────────

def build_trajectory(record: dict, cfg: Config = Config) -> np.ndarray:
    """
    Construct the multi-dimensional prosodic trajectory γ(t) ∈ ℝ^d.

    Each dimension is independently z-score normalised before stacking,
    ensuring that no single feature dominates the distance metric due to
    scale differences.

    Parameters
    ----------
    record : dict with keys 'waveform', 'label'

    Returns
    -------
    traj : np.ndarray, shape (T, d)  — raw (non-resampled) trajectory
    """
    y    = record["waveform"]
    sr   = cfg.SR
    hop  = cfg.HOP_LENGTH

    energy = extract_energy(y, hop=hop)
    pause  = get_pause_signal(y, sr=sr, hop=hop)

    # Align lengths (rounding differences)
    T = min(len(energy), len(pause))
    energy, pause = energy[:T], pause[:T]

    features = [energy, pause]

    if cfg.USE_PITCH:
        pitch = extract_pitch(y, sr=sr, hop=hop)
        pitch = pitch[:T]
        features.append(pitch)

    if cfg.USE_RATE:
        rate = extract_speech_rate(y, sr=sr, hop=hop)
        rate = rate[:T]
        features.append(rate)

    # Stack → (T, d)
    traj = np.stack(features, axis=-1).astype(np.float64)

    # Per-feature z-score normalisation (robustly, ignoring zero-variance)
    for j in range(traj.shape[1]):
        col = traj[:, j]
        mu, sigma = col.mean(), col.std()
        if sigma > 1e-8:
            traj[:, j] = (col - mu) / sigma
        else:
            traj[:, j] = 0.0

    return traj


def resample_trajectory(traj: np.ndarray, n: int = Config.TRAJ_LENGTH
                        ) -> np.ndarray:
    """
    Resample a trajectory of arbitrary length to exactly n frames using
    cubic spline interpolation along the time axis.

    Shape invariance under re-parameterisation is later handled by the
    SRVF transform, but having equal-length arrays is required for
    element-wise numpy operations.
    """
    T, d   = traj.shape
    t_orig = np.linspace(0, 1, T)
    t_new  = np.linspace(0, 1, n)
    resampled = np.zeros((n, d))
    for j in range(d):
        cs = interpolate.CubicSpline(t_orig, traj[:, j])
        resampled[:, j] = cs(t_new)
    return resampled


def extract_all_features(records: list, cfg: Config = Config) -> list:
    """
    Run the full feature extraction pipeline on a list of audio records.
    Appends 'trajectory' (raw) and 'trajectory_rs' (resampled) to each record.
    """
    print("\n── Feature extraction ───────────────────────────────────────────")
    for rec in records:
        traj       = build_trajectory(rec, cfg)
        traj_rs    = resample_trajectory(traj, n=cfg.TRAJ_LENGTH)
        rec["trajectory"]    = traj
        rec["trajectory_rs"] = traj_rs
        T_sec = len(rec["waveform"]) / cfg.SR
        print(f"  {rec['label']:30s}  raw T={len(traj):5d}  "
              f"dur={T_sec:.2f}s  d={traj.shape[1]}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GEOMETRIC REPRESENTATION (SRVF)
# ══════════════════════════════════════════════════════════════════════════════

def compute_srvf(traj: np.ndarray) -> np.ndarray:
    """
    Compute the Square-Root Velocity Function (SRVF) of a trajectory.

    Definition
    ----------
    Given a curve γ : [0,1] → ℝ^d, its velocity is γ'(t).
    The SRVF is defined as:

        q(t) = γ'(t) / √‖γ'(t)‖

    This representation has three key properties relevant to our pipeline:
    1. Re-parameterisation invariance: The elastic distance between two
       curves equals the L² distance between their SRVFs on the unit sphere,
       after optimal re-parameterisation (dynamic programming).
    2. Unit-sphere geometry: ‖q‖_{L²} is invariant to re-parameterisation,
       so the space of SRVFs is the unit Hilbert sphere S∞.
    3. Numerical stability: Division by √‖γ'‖ rather than ‖γ'‖ avoids
       amplifying noise at low-velocity frames.

    Parameters
    ----------
    traj : np.ndarray, shape (n, d)
        A resampled, normalised trajectory.

    Returns
    -------
    q : np.ndarray, shape (n, d)
        The SRVF of traj. ‖q‖_F ≈ √n in the discrete approximation.
    """
    n, d = traj.shape
    dt   = 1.0 / (n - 1)

    # Central differences for interior points; forward/backward at ends
    dg   = np.gradient(traj, dt, axis=0)          # shape (n, d)

    # Speed at each frame: ‖γ'(t)‖
    speed = np.linalg.norm(dg, axis=1, keepdims=True)  # (n, 1)

    # SRVF: q(t) = γ'(t) / √speed,  with numerical stabiliser
    eps   = 1e-8
    q     = dg / np.sqrt(speed + eps)

    return q


def srvf_l2_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Baseline L² distance between two SRVFs of the same shape (n, d).

    This approximates the elastic distance without optimal re-parameterisation.
    For a true elastic (Fisher–Rao) distance, one would additionally
    solve for the optimal diffeomorphism φ* of [0,1] via dynamic programming
    (see Tucker et al., 2013). That step is optional; the baseline already
    captures gross shape differences relevant to hesitation profiling.

    d(q1, q2) = (1/n) Σ_t ‖q1(t) - q2(t)‖²  [discrete approximation]
    """
    assert q1.shape == q2.shape, "SRVFs must have identical shapes."
    diff = q1 - q2
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def geodesic_distance_approx(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Approximate geodesic distance on the unit Hilbert sphere S∞.

    The geodesic (arc-length) between two unit vectors u, v on a sphere is
        d_geo = arccos(⟨u, v⟩)
    For SRVFs in discrete form, ⟨q1, q2⟩_L² = (1/n) Σ_t q1(t)·q2(t).

    This is a closer approximation to the true Fisher–Rao distance than the
    chord distance (L² distance) above, assuming no re-parameterisation is
    needed. For publication-quality results, use optimal re-parameterisation
    (DP alignment) and then compute this arc-length.
    """
    n = q1.shape[0]
    # Discrete L² inner product
    inner = np.sum(q1 * q2) / n

    # Normalise SRVFs onto the unit sphere
    norm1 = np.sqrt(np.sum(q1 ** 2) / n)
    norm2 = np.sqrt(np.sum(q2 ** 2) / n)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cos_theta = inner / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # guard arccos domain
    return float(np.arccos(cos_theta))


def dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance as an alternative elastic alignment.

    DTW finds the optimal monotone alignment between two sequences,
    minimising total point-wise distance. Unlike SRVF geodesic, it does not
    require equal arc-length normalisation, but it is also not a proper
    Riemannian metric.

    This is provided as an experimental baseline for comparison.
    """
    n, m  = len(traj1), len(traj2)
    cost  = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.sum((traj1[i - 1] - traj2[j - 1]) ** 2)
            cost[i, j] = d + min(cost[i - 1, j],
                                  cost[i, j - 1],
                                  cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def compute_srvfs(records: list, cfg: Config = Config) -> list:
    """
    Compute SRVFs for all records.
    Appends 'srvf' to each record dict.
    """
    print("\n── SRVF computation ─────────────────────────────────────────────")
    for rec in records:
        rec["srvf"] = compute_srvf(rec["trajectory_rs"])
    print(f"  SRVF shape: {records[0]['srvf'].shape}  "
          f"(frames × features)")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PAIRWISE DISTANCE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def compute_distance_matrix(records: list,
                             metric: str = "srvf_l2") -> np.ndarray:
    """
    Compute a symmetric pairwise distance matrix over all records.

    Parameters
    ----------
    metric : 'srvf_l2'  — baseline L² between SRVFs
             'geodesic' — arc-length on S∞ (approximate)
             'dtw'      — dynamic time warping on raw trajectories

    Returns
    -------
    D : np.ndarray, shape (N, N)
    """
    N    = len(records)
    D    = np.zeros((N, N))

    metric_fn = {
        "srvf_l2":  lambda r1, r2: srvf_l2_distance(r1["srvf"], r2["srvf"]),
        "geodesic": lambda r1, r2: geodesic_distance_approx(r1["srvf"],
                                                             r2["srvf"]),
        "dtw":      lambda r1, r2: dtw_distance(r1["trajectory_rs"],
                                                r2["trajectory_rs"]),
    }[metric]

    print(f"\n── Pairwise distances ({metric}) ──────────────────────────────────")
    for i in range(N):
        for j in range(i + 1, N):
            d       = metric_fn(records[i], records[j])
            D[i, j] = d
            D[j, i] = d
        if i % max(1, N // 5) == 0:
            print(f"  Row {i+1}/{N} done …")

    print(f"  Distance matrix: shape={D.shape}, "
          f"mean={D.mean():.4f}, std={D.std():.4f}")
    return D


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRADITIONAL FLUENCY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_fluency_metrics(record: dict, cfg: Config = Config) -> dict:
    """
    Compute classical speech fluency metrics for correlation analysis.

    These are the operationalisations used in Cucchiarini et al. (2002) and
    Skehan (1998):
    • Total pause duration  — total time (seconds) in detected pauses
    • Pause count           — number of distinct pause events
    • Speech rate           — voiced frames per second (proxy)
    """
    pause  = record["trajectory"][:, 1]       # raw pause signal
    sr_f   = cfg.SR / cfg.HOP_LENGTH           # frames per second

    # Total pause duration
    total_pause_dur = float(np.sum(pause)) / sr_f

    # Pause count: count rising edges
    diff          = np.diff(np.concatenate([[0], (pause > 0.5).astype(int)]))
    pause_count   = int(np.sum(diff == 1))

    # Speech rate: fraction of non-pause frames per second
    T_sec         = len(record["waveform"]) / cfg.SR
    voiced_frac   = float(np.sum(pause < 0.5)) / len(pause)
    speech_rate   = voiced_frac / max(T_sec, 1e-3)   # voiced proportion / sec

    return {
        "total_pause_dur": total_pause_dur,
        "pause_count":     pause_count,
        "speech_rate":     speech_rate,
    }


def add_fluency_metrics(records: list, cfg: Config = Config) -> list:
    for rec in records:
        rec["fluency"] = compute_fluency_metrics(rec, cfg)
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 7A. Fréchet-inspired centroid ─────────────────────────────────────────────

def compute_srvf_centroid(records: list) -> np.ndarray:
    """
    Compute an approximate Fréchet mean in SRVF space.

    The true Fréchet mean on S∞ minimises Σ d²(q_i, μ). Computing it
    requires iterative geodesic shooting (Tucker et al., 2013). Here we
    use the normalised arithmetic mean as a fast, consistent approximation
    (it converges to the true mean for small dispersions — valid when
    within-group variance is low, as expected for native speakers).

    For publication: replace with the iterative Karcher mean algorithm.
    """
    srvfs  = np.stack([r["srvf"] for r in records], axis=0)  # (N, n, d)
    mean_q = srvfs.mean(axis=0)                               # (n, d)
    # Project back to the sphere (optional normalisation)
    norm   = np.sqrt(np.mean(mean_q ** 2) + 1e-12)
    return mean_q / norm


def deviation_from_centroid(record: dict, centroid: np.ndarray,
                             metric: str = "srvf_l2") -> float:
    """
    Compute a speaker's hesitation score as their SRVF distance
    from the native centroid.

    Large deviation → high hesitation / non-nativeness.
    """
    if metric == "geodesic":
        return geodesic_distance_approx(record["srvf"], centroid)
    else:
        return srvf_l2_distance(record["srvf"], centroid)


# ── 7B. Intra / inter group distances ────────────────────────────────────────

def group_distances(D: np.ndarray, labels: list) -> dict:
    """
    Extract L1–L1, L2–L2, L1–L2 off-diagonal distances.
    """
    N     = len(labels)
    l1_i  = [i for i, r in enumerate(labels) if r == "L1"]
    l2_i  = [i for i, r in enumerate(labels) if r == "L2"]

    def _pairs(idx_a, idx_b, symmetric=True):
        dists = []
        for a in idx_a:
            for b in idx_b:
                if symmetric and b <= a:
                    continue
                if a != b:
                    dists.append(D[a, b])
        return np.array(dists)

    return {
        "L1-L1": _pairs(l1_i, l1_i),
        "L2-L2": _pairs(l2_i, l2_i),
        "L1-L2": _pairs(l1_i, l2_i, symmetric=False),
    }


def statistical_tests(group_dist: dict, alpha: float = Config.ALPHA):
    """
    Welch's t-test (unequal variances) for all pairs of distance groups,
    plus Mann–Whitney U test as a non-parametric companion.
    """
    print("\n── Statistical tests ────────────────────────────────────────────")
    keys   = list(group_dist.keys())
    for k1, k2 in itertools.combinations(keys, 2):
        a, b = group_dist[k1], group_dist[k2]
        if len(a) < 2 or len(b) < 2:
            print(f"  {k1} vs {k2}: insufficient samples.")
            continue
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        sig    = "***" if p_t < 0.001 else "**" if p_t < 0.01 else \
                 "*" if p_t < alpha else "ns"
        print(f"  {k1} vs {k2}:  "
              f"t={t:+.3f} p_t={p_t:.4f} {sig}  |  "
              f"U={u:.0f} p_u={p_u:.4f}")
        print(f"    means: {k1}={a.mean():.4f}  {k2}={b.mean():.4f}")


# ── 7C. Correlation with classical metrics ───────────────────────────────────

def correlation_analysis(records: list, centroid: np.ndarray,
                          metric: str = "srvf_l2") -> dict:
    """
    Pearson and Spearman correlation between the geometric hesitation score
    and traditional fluency metrics.

    A strong correlation validates the geometric distance as a fluency
    measure; it also shows that the Riemannian representation captures the
    same information as (but potentially more than) classical metrics.
    """
    geom_scores = np.array([
        deviation_from_centroid(r, centroid, metric) for r in records
    ])
    trad_keys = ["total_pause_dur", "pause_count", "speech_rate"]
    results   = {}
    print("\n── Correlation (geometric score vs traditional metrics) ──────────")
    for key in trad_keys:
        vals = np.array([r["fluency"][key] for r in records])
        if vals.std() < 1e-8:
            print(f"  {key}: zero variance, skipped.")
            continue
        r_p, p_p = stats.pearsonr(geom_scores, vals)
        r_s, p_s = stats.spearmanr(geom_scores, vals)
        results[key] = {"pearson": (r_p, p_p), "spearman": (r_s, p_s),
                        "vals": vals, "scores": geom_scores}
        print(f"  {key:22s}: Pearson r={r_p:+.3f} (p={p_p:.4f})  "
              f"Spearman ρ={r_s:+.3f} (p={p_s:.4f})")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _colors(records, cfg=Config):
    return [cfg.COLOR_L1 if r["group"] == "L1" else cfg.COLOR_L2
            for r in records]


def plot_embedding(D: np.ndarray, records: list,
                   output_dir: Path, cfg: Config = Config,
                   method: str = "MDS"):
    """
    Project the pairwise distance matrix into 2D using MDS or PCA.

    MDS preserves distances as faithfully as possible in 2D.
    PCA operates on the (centred) distance matrix as a kernel (PCoA).

    The resulting scatter plot reveals whether L1 and L2 clusters are
    geometrically separable in SRVF space — a key result for the paper.
    """
    if method == "MDS":
        emb = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, max_iter=1000, n_init=10)
        coords = emb.fit_transform(D)
        stress = emb.stress_
        title  = f"MDS Embedding of SRVF Distances  (stress={stress:.3f})"
        xlabel, ylabel = "MDS Dim 1", "MDS Dim 2"
    else:   # PCA on centred distance matrix (PCoA)
        H      = np.eye(len(D)) - np.ones_like(D) / len(D)
        B      = -0.5 * H @ (D ** 2) @ H
        B      = (B + B.T) / 2
        vals, vecs = np.linalg.eigh(B)
        idx       = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        vals_pos  = np.maximum(vals[:2], 0)
        coords    = vecs[:, :2] * np.sqrt(vals_pos)
        pve       = vals_pos / np.maximum(vals[vals > 0].sum(), 1e-12) * 100
        title     = "PCoA Embedding of SRVF Distances"
        xlabel    = f"PC 1 ({pve[0]:.1f}% var)"
        ylabel    = f"PC 2 ({pve[1]:.1f}% var)"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors  = _colors(records, cfg)

    for i, rec in enumerate(records):
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=colors[i], s=80, edgecolors="white",
                   linewidths=0.6, zorder=3)
        ax.annotate(rec["label"].split("_", 1)[-1],
                    xy=(coords[i, 0], coords[i, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=colors[i], alpha=0.85)

    # Group convex hulls
    for grp, col in [("L1", cfg.COLOR_L1), ("L2", cfg.COLOR_L2)]:
        pts = coords[[i for i, r in enumerate(records) if r["group"] == grp]]
        if len(pts) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        alpha=0.08, color=col)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=col, linewidth=1.2, linestyle="--", alpha=0.6)
            except Exception:
                pass

    legend_patches = [
        mpatches.Patch(color=cfg.COLOR_L1, label="Native (L1)"),
        mpatches.Patch(color=cfg.COLOR_L2, label="L2 Learners"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, frameon=False)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    fig.tight_layout()
    out = output_dir / f"embedding_{method.lower()}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_group_boxplots(group_dist: dict, output_dir: Path,
                        cfg: Config = Config):
    """
    Boxplot comparing L1–L1, L2–L2, and L1–L2 pairwise distances.

    Wider within-group spread for L2 means more heterogeneous prosodic
    trajectories — consistent with variable hesitation patterns.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    data    = [group_dist[k] for k in ["L1-L1", "L2-L2", "L1-L2"]]
    labels  = ["L1–L1", "L2–L2", "L1–L2"]
    colors  = [cfg.COLOR_L1, cfg.COLOR_L2, "#6A994E"]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4,
                                    markeredgecolor="grey", alpha=0.6))

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("SRVF Distance", fontsize=10)
    ax.set_title("Pairwise SRVF Distances by Group Pair", pad=10)

    # Significance brackets
    def _bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black")
        ax.text((x1 + x2) / 2, y + h, text, ha="center",
                va="bottom", fontsize=9)

    tops = [np.percentile(d, 95) for d in data]
    ymax = max(tops) * 1.05
    if len(data[0]) >= 2 and len(data[1]) >= 2:
        _, p = stats.ttest_ind(data[0], data[1], equal_var=False)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        _bracket(ax, 1, 2, ymax, ymax * 0.03, sig)

    fig.tight_layout()
    out = output_dir / "boxplots_group_distances.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_deviation_scores(records: list, centroid: np.ndarray,
                          output_dir: Path, cfg: Config = Config,
                          metric: str = "srvf_l2"):
    """
    Strip plot of hesitation scores (distance to native centroid)
    for L1 and L2 speakers.

    The centroid is computed from L1 speakers only. Each dot is a speaker;
    vertical spread is jittered for readability. A larger score is
    interpreted as greater geometric deviation from the native prosodic
    prototype — our operationalisation of hesitation.
    """
    scores = {
        "L1": [(r["label"], deviation_from_centroid(r, centroid, metric))
               for r in records if r["group"] == "L1"],
        "L2": [(r["label"], deviation_from_centroid(r, centroid, metric))
               for r in records if r["group"] == "L2"],
    }

    fig, ax = plt.subplots(figsize=(5.5, 5))
    rng    = np.random.default_rng(0)

    for xi, (grp, col) in enumerate([("L1", cfg.COLOR_L1),
                                      ("L2", cfg.COLOR_L2)], 1):
        vals = np.array([s[1] for s in scores[grp]])
        jit  = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), xi) + jit, vals,
                   c=col, s=70, edgecolors="white", linewidths=0.5,
                   zorder=3, alpha=0.85)
        ax.hlines(np.median(vals), xi - 0.25, xi + 0.25,
                  colors=col, linewidths=2.5, zorder=4)
        for lbl, v in scores[grp]:
            ax.annotate(lbl.split("_", 1)[-1],
                        xy=(xi + jit[scores[grp].index((lbl, v))], v),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=6, color=col, alpha=0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Native (L1)", "L2 Learners"], fontsize=10)
    ax.set_ylabel("Geometric Hesitation Score\n(distance to L1 centroid)",
                  fontsize=10)
    ax.set_title("SRVF Deviation from Native Centroid", pad=10)
    ax.set_xlim(0.5, 2.7)

    fig.tight_layout()
    out = output_dir / "hesitation_scores.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_correlations(corr_results: dict, records: list,
                      output_dir: Path, cfg: Config = Config):
    """
    Scatter plots: geometric hesitation score vs each traditional metric.

    Include regression line and 95 % confidence band.
    If geometric scores correlate strongly with total pause duration and
    pause count but less so with speech rate, this suggests that the
    SRVF distance specifically captures pause-related hesitation.
    """
    n_plots = len(corr_results)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
    colors = _colors(records, cfg)

    for ax, (key, res) in zip(axes, corr_results.items()):
        x = res["scores"]
        y = res["vals"]
        r_p, p_p = res["pearson"]

        ax.scatter(x, y, c=colors, s=65, edgecolors="white",
                   linewidths=0.5, zorder=3, alpha=0.85)

        # Regression line + 95% confidence band
        slope, intercept, *_ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#333333", linewidth=1.5, zorder=2)

        # CI band via bootstrap (simple approach)
        n  = len(x)
        se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2) /
                     np.sum((x - x.mean()) ** 2))
        t_crit = stats.t.ppf(0.975, df=n - 2)
        x_c    = (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
        y_se   = np.sqrt((1 / n + x_c)) * se * np.sqrt(n)
        ax.fill_between(x_line, y_line - t_crit * y_se,
                        y_line + t_crit * y_se,
                        alpha=0.12, color="#333333")

        label_map = {
            "total_pause_dur": "Total Pause Duration (s)",
            "pause_count":     "Pause Count",
            "speech_rate":     "Speech Rate (voiced/s)",
        }
        ax.set_xlabel("Geometric Hesitation Score", fontsize=9)
        ax.set_ylabel(label_map.get(key, key), fontsize=9)
        ax.set_title(f"r = {r_p:+.3f}  (p = {p_p:.4f})", fontsize=9, pad=8)

    legend_patches = [
        mpatches.Patch(color=cfg.COLOR_L1, label="L1"),
        mpatches.Patch(color=cfg.COLOR_L2, label="L2"),
    ]
    fig.legend(handles=legend_patches, loc="lower right",
               fontsize=8, frameon=False)
    fig.suptitle("Geometric Score vs Traditional Fluency Metrics",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = output_dir / "correlation_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_example_trajectories(records: list, output_dir: Path,
                               cfg: Config = Config, n_each: int = 3):
    """
    Overlay the resampled trajectories for a few L1 and L2 speakers,
    feature by feature. Visually demonstrates the greater irregularity
    of L2 trajectories — especially in the pause and energy channels.
    """
    feature_names = ["Energy (dB)", "Pause Signal"]
    if cfg.USE_PITCH:  feature_names.append("Pitch (norm.)")
    if cfg.USE_RATE:   feature_names.append("Speech Rate")
    d    = len(feature_names)
    t    = np.linspace(0, 1, cfg.TRAJ_LENGTH)

    l1_sub = [r for r in records if r["group"] == "L1"][:n_each]
    l2_sub = [r for r in records if r["group"] == "L2"][:n_each]

    fig, axes = plt.subplots(d, 1, figsize=(8, 2.5 * d), sharex=True)
    if d == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        for rec in l1_sub:
            ax.plot(t, rec["trajectory_rs"][:, j],
                    color=cfg.COLOR_L1, linewidth=1.0, alpha=0.6)
        for rec in l2_sub:
            ax.plot(t, rec["trajectory_rs"][:, j],
                    color=cfg.COLOR_L2, linewidth=1.0, alpha=0.6,
                    linestyle="--")
        ax.set_ylabel(feature_names[j], fontsize=9)
        ax.axhline(0, color="grey", lw=0.4, alpha=0.4)

    axes[-1].set_xlabel("Normalised Time", fontsize=9)
    legend_patches = [
        mpatches.Patch(color=cfg.COLOR_L1, label="L1 (solid)"),
        mpatches.Patch(color=cfg.COLOR_L2, label="L2 (dashed)"),
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               fontsize=8, frameon=False)
    fig.suptitle("Prosodic Trajectories: L1 vs L2 Overlay", fontsize=11)
    fig.tight_layout()
    out = output_dir / "trajectories_overlay.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_distance_heatmap(D: np.ndarray, records: list,
                          output_dir: Path, cfg: Config = Config):
    """
    Colour-coded pairwise distance heatmap (sorted by group).
    Off-diagonal blocks reveal L1–L2 separation at a glance.
    """
    # Sort so L1 come first
    idx   = sorted(range(len(records)),
                   key=lambda i: (records[i]["group"], records[i]["label"]))
    D_s   = D[np.ix_(idx, idx)]
    lbls  = [records[i]["label"].split("_", 1)[-1] for i in idx]
    grps  = [records[i]["group"] for i in idx]

    fig, ax = plt.subplots(figsize=(max(5, len(records) * 0.55 + 1),
                                     max(4, len(records) * 0.55)))
    im = ax.imshow(D_s, aspect="auto",
                   cmap="RdYlBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="SRVF Distance", fraction=0.04, pad=0.03)

    ax.set_xticks(range(len(lbls)))
    ax.set_yticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(lbls, fontsize=7)

    # Separator line between L1 / L2 blocks
    n_l1 = sum(1 for g in grps if g == "L1")
    if 0 < n_l1 < len(grps):
        ax.axhline(n_l1 - 0.5, color="white", linewidth=2)
        ax.axvline(n_l1 - 0.5, color="white", linewidth=2)
        ax.text(n_l1 / 2, -1.0, "L1", ha="center", va="bottom",
                fontsize=9, color=cfg.COLOR_L1, fontweight="bold")
        ax.text(n_l1 + (len(grps) - n_l1) / 2, -1.0, "L2",
                ha="center", va="bottom", fontsize=9,
                color=cfg.COLOR_L2, fontweight="bold")

    ax.set_title("Pairwise SRVF Distance Matrix", pad=10)
    fig.tight_layout()
    out = output_dir / "distance_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SYNTHETIC DATA GENERATOR (for testing without real audio)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(l1_dir: Path, l2_dir: Path,
                                n_l1: int = 8, n_l2: int = 8,
                                sr: int = Config.SR, dur: float = 5.0):
    """
    Generate synthetic .wav files for pipeline testing.

    L1 speakers: voiced sinusoid with sparse, short pauses.
    L2 speakers: same but with longer, more frequent pauses and
                 added low-frequency pitch perturbation.

    This is NOT a phonetically valid simulation — it is only a smoke-test
    to verify the pipeline runs without real data. Replace with actual
    recordings for research use.
    """
    l1_dir.mkdir(parents=True, exist_ok=True)
    l2_dir.mkdir(parents=True, exist_ok=True)

    rng  = np.random.default_rng(2024)
    n    = int(sr * dur)
    t    = np.linspace(0, dur, n)

    def _voiced_segment(length, f0=150.0, noise=0.05):
        return (np.sin(2 * np.pi * f0 * np.linspace(0, 1, length)) +
                noise * rng.standard_normal(length)).astype(np.float32)

    def _with_pauses(length, pause_prob=0.1, pause_dur_range=(0.1, 0.3)):
        y   = _voiced_segment(length)
        pos = 0
        while pos < length:
            if rng.random() < pause_prob:
                plen = int(rng.uniform(*pause_dur_range) * sr)
                y[pos: pos + plen] = 0.0
                pos += plen
            else:
                pos += int(rng.uniform(0.05, 0.3) * sr)
        return y

    import soundfile as sf

    for i in range(n_l1):
        y = _with_pauses(n, pause_prob=0.06,
                          pause_dur_range=(0.05, 0.15))
        sf.write(str(l1_dir / f"native_{i+1:02d}.wav"), y, sr)

    for i in range(n_l2):
        y = _with_pauses(n, pause_prob=0.20,
                          pause_dur_range=(0.15, 0.40))
        sf.write(str(l2_dir / f"learner_{i+1:02d}.wav"), y, sr)

    print(f"  Synthetic data: {n_l1} L1 + {n_l2} L2 files written.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(records, D, group_dist, centroid, cfg=Config):
    n_l1  = sum(1 for r in records if r["group"] == "L1")
    n_l2  = sum(1 for r in records if r["group"] == "L2")
    print("\n" + "═" * 66)
    print("  PIPELINE SUMMARY")
    print("═" * 66)
    print(f"  Speakers  : {n_l1} L1, {n_l2} L2  (total {n_l1+n_l2})")
    print(f"  Features  : dim={cfg.feature_dim()}, "
          f"T={cfg.TRAJ_LENGTH} (resampled)")
    print(f"  Distance  : srvf_l2  (mean = {D.mean():.4f}, "
          f"std = {D.std():.4f})")
    for grp, arr in group_dist.items():
        if len(arr):
            print(f"  {grp:8s}: mean={arr.mean():.4f} ± {arr.std():.4f}  "
                  f"(n={len(arr)} pairs)")

    l1_recs = [r for r in records if r["group"] == "L1"]
    l2_recs = [r for r in records if r["group"] == "L2"]
    if l1_recs and l2_recs:
        l1_dev = np.mean([deviation_from_centroid(r, centroid) for r in l1_recs])
        l2_dev = np.mean([deviation_from_centroid(r, centroid) for r in l2_recs])
        print(f"  Deviation from L1 centroid : "
              f"L1={l1_dev:.4f}  L2={l2_dev:.4f}")
    print("═" * 66 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: Config = Config, use_synthetic: bool = True,
                 distance_metric: str = "srvf_l2"):
    """
    Execute the full research pipeline end-to-end.

    Parameters
    ----------
    cfg              : Config object
    use_synthetic    : if True, generate synthetic .wav files first
                       (useful for testing without real data)
    distance_metric  : 'srvf_l2' | 'geodesic' | 'dtw'
    """
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. (Optional) generate synthetic data ─────────────────────────────────
    if use_synthetic:
        print("\n── Generating synthetic test data ───────────────────────────────")
        generate_synthetic_dataset(cfg.L1_DIR, cfg.L2_DIR)

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    l1_records, l2_records = load_dataset(cfg.L1_DIR, cfg.L2_DIR, sr=cfg.SR)
    all_records = l1_records + l2_records

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    all_records = extract_all_features(all_records, cfg)
    all_records = add_fluency_metrics(all_records, cfg)

    # ── 3. SRVF computation ───────────────────────────────────────────────────
    all_records = compute_srvfs(all_records, cfg)

    # ── 4. Pairwise distance matrix ───────────────────────────────────────────
    D = compute_distance_matrix(all_records, metric=distance_metric)

    # ── 5. Group-level distances ──────────────────────────────────────────────
    labels     = [r["group"] for r in all_records]
    group_dist = group_distances(D, labels)
    statistical_tests(group_dist, alpha=cfg.ALPHA)

    # ── 6. Native centroid & hesitation scores ────────────────────────────────
    l1_records_only = [r for r in all_records if r["group"] == "L1"]
    centroid        = compute_srvf_centroid(l1_records_only)

    # ── 7. Correlation analysis ───────────────────────────────────────────────
    corr_results = correlation_analysis(all_records, centroid,
                                         metric=distance_metric)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print_summary(all_records, D, group_dist, centroid, cfg)

    # ── 9. Visualisation ──────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────")
    plot_example_trajectories(all_records, cfg.OUTPUT_DIR, cfg)
    plot_distance_heatmap(D, all_records, cfg.OUTPUT_DIR, cfg)
    plot_group_boxplots(group_dist, cfg.OUTPUT_DIR, cfg)
    plot_deviation_scores(all_records, centroid, cfg.OUTPUT_DIR, cfg,
                          metric=distance_metric)
    plot_correlations(corr_results, all_records, cfg.OUTPUT_DIR, cfg)
    plot_embedding(D, all_records, cfg.OUTPUT_DIR, cfg, method="MDS")
    plot_embedding(D, all_records, cfg.OUTPUT_DIR, cfg, method="PCoA")

    print(f"\n✓ Pipeline complete. Figures saved to: {cfg.OUTPUT_DIR.resolve()}")
    return {
        "records":    all_records,
        "D":          D,
        "group_dist": group_dist,
        "centroid":   centroid,
        "corr":       corr_results,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline(
    cfg=Config,
    use_synthetic=False,   # ← change to False
    distance_metric="srvf_l2"
)
