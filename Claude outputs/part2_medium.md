# Beyond the Scale, Part 2: Generating Fake Fish
## GaussianCopula, CTGAN and TVAE Applied to Aquaculture Morphometrics

*Part 2 of a 3-part series on synthetic data for allometric weight estimation in precision aquaculture.*  
*← [Part 1: The Data Challenge, Biometrics & Allometric Baselines](#)*

---

## 1. Where We Left Off — and What Comes Next

In Part 1 we established two things. First, the morphometric dataset for *Solea senegalensis* is small by necessity — 209 individuals, ethically constrained to a single measurement session — and yet the multivariate allometric model W = a·L^b₁·A^b₂·E^b₃ already achieves MAE = 0.352 g and R² = 0.973 in 5-fold cross-validation on the training set. That is a strong baseline.

Second, and more importantly, we identified the two structural gaps that the baseline cannot address on its own: the upper weight tail is sparsely sampled (fewer than ~10% of individuals exceed 12 g). In this context — juvenile redistribution into homogeneous growth tanks — grading is performed by size category, not by market weight. Weight estimation serves as a proxy for morphometric classification. Misclassifying the largest individuals matters not because of their immediate commercial value, but because placing them alongside significantly smaller juveniles reintroduces the size heterogeneity that grading was designed to eliminate, reigniting competitive dominance and growth divergence in the destination tank, and the model has never seen variation beyond a single cohort.

These are the two scenarios where synthetic data augmentation is most commonly proposed. The logic is intuitive: if we can learn the joint distribution of {length, width, thickness, weight} from the 167 training observations and sample new points from it, we could potentially expand the effective training set, improve tail coverage, and make the weight estimator more robust.

But intuition is not evidence. The question this series is designed to answer is more precise: **do synthetic observations generated from a small real dataset actually improve weight estimation on real, unseen fish — and if so, under what conditions?**

Part 2 builds the machinery to answer that question. We train three generative models on the real training data, evaluate whether the synthetic fish they produce are statistically faithful and biologically credible, and compute a preliminary Train on Synthetic, Test on Real (TSTR) benchmark against the baselines established in Part 1. Part 3 will complete the comparison across the full range of augmentation strategies, algorithms and proportions.

All code is in the companion notebook **`part2_synthetic_generation.ipynb`**.

---

## 2. Three Ways to Learn a Distribution from Fish

Generating synthetic tabular data is not a single problem — it is a family of problems with different assumptions, failure modes and computational costs. We evaluate three models from the [Synthetic Data Vault (SDV)](https://sdv.dev/) library, each representing a distinct methodological approach.

### 2.1 GaussianCopula — The Probabilistic Baseline

A copula is a mathematical function that separates the marginal distributions of individual variables from the dependence structure between them. The GaussianCopula approach proceeds in two steps: first transform each variable so that it follows a standard normal distribution (using a parametric or non-parametric transformation); then fit a multivariate Gaussian to the transformed variables, capturing the correlation structure in a single covariance matrix.

Sampling reverses the process: draw from the multivariate Gaussian, then invert the transformations to recover samples in the original scale.

**Why it matters for small datasets:** The GaussianCopula has very few parameters — essentially the marginal transformations plus a correlation matrix. With n = 167, this is an advantage: a model with few parameters has less opportunity to overfit or memorise. The cost is the Gaussian assumption on the dependence structure, which may miss nonlinear correlations or multimodal joint distributions.

**What to expect:** Good reproduction of marginal distributions and linear correlations. Potential underperformance in the tails and in capturing the nonlinear allometric structure. Computationally fast and deterministic given a fixed seed.

> **Deeper dive:** Sklar's theorem (1959) guarantees that any multivariate distribution can be expressed as a copula applied to its marginals — making the GaussianCopula a theoretically principled, not ad hoc, approach. See [Nelson (2006)](https://link.springer.com/book/10.1007/0-387-28678-0) for a rigorous treatment.

### 2.2 CTGAN — The Adversarial Approach

CTGAN (Conditional Tabular GAN) adapts the Generative Adversarial Network framework to tabular data. A generator network learns to produce fake rows that fool a discriminator network into classifying them as real. The two networks are trained simultaneously in a minimax game: the generator improves by fooling the discriminator; the discriminator improves by detecting fakes.

Two key modifications make GANs viable for tabular data. First, **mode-specific normalisation**: continuous variables are modelled as Gaussian mixtures, and the generator conditions on the active mixture component — preventing the mode collapse that plagues vanilla GANs on multimodal distributions. Second, **training-by-sampling**: each training batch oversamples underrepresented modes, addressing the imbalance that arises when some regions of the feature space are sparsely populated.

**Why it matters here:** The allometric relationship W ∝ L^b₁·A^b₂·E^b₃ is nonlinear. If CTGAN can learn this nonlinear joint structure rather than just the marginals, it may generate fish whose morphometric combinations are more physically coherent than those from the GaussianCopula.

**What to expect:** Higher expressiveness but also higher variance — CTGAN is sensitive to hyperparameters, training instability and sample size. With only 167 training records, the discriminator has very limited signal. Results may vary substantially across random seeds.

> **Deeper dive:** [Xu et al. (2019)](https://arxiv.org/abs/1907.00503) — the original CTGAN paper — provides a thorough treatment of the mode-specific normalisation and conditional training strategy.

### 2.3 TVAE — The Variational Approach

TVAE (Tabular Variational Autoencoder) applies the VAE framework to tabular data. An encoder network maps each real row to a distribution in a low-dimensional latent space (the posterior); a decoder network reconstructs the original row from latent samples. Training minimises the Evidence Lower BOund (ELBO): reconstruction accuracy plus a regularisation term that keeps the posterior close to a standard Gaussian prior.

Sampling is straightforward: draw a point from the standard Gaussian prior, pass it through the decoder.

**Why it matters here:** VAEs tend to produce smoother, more interpolated samples than GANs — which is an asset when the training set is small and the goal is to fill in gaps rather than extrapolate. The regularisation term explicitly discourages memorisation, reducing the risk that synthetic samples are simple copies of training records.

**What to expect:** Better stability than CTGAN (no adversarial game) and potentially better tail coverage than the GaussianCopula. The key risk is posterior collapse — if the latent space loses structure, the decoder generates generic, low-diversity samples regardless of what prior point was drawn.

> **Deeper dive:** [Kingma & Welling (2013)](https://arxiv.org/abs/1312.6114) introduced the VAE; [Xu et al. (2019)](https://arxiv.org/abs/1907.00503) describes the tabular adaptation in TVAE.

### 2.4 Model Selection Criteria

We do not select one model before the experiment. We train all three, evaluate them on the same criteria, and let the data decide. The evaluation covers four independent dimensions:

| Dimension | What we measure |
|---|---|
| **Univariate fidelity** | Marginal distributions, moments, KS statistics |
| **Multivariate fidelity** | Correlation matrices, joint scatter structure, allometric preservation |
| **Biological validity** | Same checks as Section 4.5 of Part 1: positivity, ordering, K_area, ratios |
| **Predictive utility** | TSTR: train on synthetic, test on real — compared against TRTR baseline |

A model that scores well on the first three dimensions but poorly on predictive utility is generating statistically plausible but predictively useless fish. Conversely, a model that scores well on TSTR but poorly on fidelity may be memorising training data rather than learning the distribution. We need all four lenses.

---

## 3. The One Rule That Cannot Be Broken

Before writing a single line of synthesis code, one methodological constraint must be established explicitly — because violating it would invalidate every result that follows.

**The test set must not exist when the synthesiser is trained.**

This sounds obvious, but it is violated in a surprisingly large proportion of published studies on synthetic data augmentation. The failure mode is subtle: if the synthesiser sees the full dataset (including test records) before the train/test split, then the synthetic data it generates may contain information that "leaks" from the test set into the training process. The downstream models are then evaluated on data they have, in a probabilistic sense, already seen.

In our pipeline:

```
real_train.parquet  ──→  synthesiser.fit()  ──→  synthetic samples
real_test.parquet   ──→  sealed until Part 3 final evaluation
```

The test set (`real_test.parquet`, 20% of the original 209 records) was created in `part1_eda_v2.ipynb` before any analysis took place. It has not been used since. It will not be used in this notebook. When we compute TSTR metrics in this part, we use cross-validation on the training set — not the sealed test set.

The sealed test set appears exactly once, in Part 3, to compute the final comparison between augmentation strategies. This design ensures that every modelling decision made in Parts 1 and 2 — model selection, hyperparameter tuning, proportion of synthetic data — is made blind to the test performance.

---

## 4. The Synthesis Pipeline

### 4.1 Environment and Metadata

SDV ≥ 1.0 introduced a clean separation between the *metadata* layer (which describes the schema of your table) and the *synthesiser* layer (which learns the distribution). This separation matters: it forces you to be explicit about which columns are numeric, which are categorical, and which are identifiers that should never be synthesised. In our case the schema is simple — four continuous measurements and one traceability column — but defining it explicitly protects against silent type inference errors.

```python
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_train)

# Override: data_origin is traceability-only, not a feature to synthesise
metadata.update_column("data_origin", sdtype="categorical")

# Confirm numeric sdtypes for the four measurement columns
for col in ["weight_g", "length_cm", "width_cm", "thickness_cm"]:
    metadata.update_column(col, sdtype="numerical")
```

> **Note on `data_origin`:** this column was added in Part 1 to tag every record as `"real_train"`, `"real_test"` or `"synthetic"`. It must never be fed to a predictive model as a feature — doing so would constitute data leakage in a different form. Declare it as categorical in the metadata so that SDV does not learn its marginal distribution.

### 4.2 Fitting the Three Synthesisers

Each synthesiser receives the same training data (`real_train`) and the same metadata object. Hyperparameters are intentionally kept at sensible defaults for this first pass — tuning will be revisited in Part 3 if the evaluation warrants it.

```python
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)

RANDOM_STATE = 42

synthesisers = {
    "gaussian_copula": GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
        default_distribution="norm",
    ),
    "ctgan": CTGANSynthesizer(
        metadata,
        epochs=500,
        batch_size=50,          # ≈ n/3; small batches necessary with n=167
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=False,
        cuda=False,
    ),
    "tvae": TVAESynthesizer(
        metadata,
        epochs=500,
        batch_size=50,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
    ),
}

# Fit — only on real_train, never on real_test
fitted = {}
for name, synth in synthesisers.items():
    synth.fit(df_train.drop(columns=["data_origin"]))
    fitted[name] = synth
    print(f"  ✓ {name} fitted")
```

A few design decisions deserve explanation:

**`batch_size=50` for CTGAN and TVAE.** This parameter controls the number of *real training records* fed to the network in each gradient update step — it has nothing to do with the number of synthetic samples generated (which is set separately via `num_rows` in `.sample()`). With n = 167 training records, the standard default of 500 would mean each mini-batch is larger than the entire dataset, reducing every epoch to a single gradient step. Setting `batch_size=50` gives approximately three updates per epoch (167 / 50 ≈ 3), providing a meaningful optimisation signal without trivialising training.

**`epochs=500`.** With a small dataset and small batch size, 500 epochs provides a reasonable training budget without excessive runtime. Convergence should be monitored via the training loss, which the notebook exposes.

**`enforce_min_max_values=True` for GaussianCopula.** This clips any sample outside the observed range of each variable. It is a coarse safety net, not a substitute for the biological validity checks in §4.4.

**`cuda=False`.** The dataset is small enough that CPU training is faster than the GPU overhead for CTGAN and TVAE.

### 4.3 Generating Multiple Replicas

A single synthetic sample is not a reliable estimator of anything. Aleatoric variability in the generator — particularly for CTGAN, which has a stochastic discriminator — means that two runs with different seeds may produce materially different distributions. We generate **R = 5 independent replicas** per synthesiser at each sample size, using fixed seeds for reproducibility.

```python
import pandas as pd

N_REAL_TRAIN = len(df_train)   # 167
PROPORTIONS  = [0.25, 0.50, 1.00, 2.00]   # relative to N_REAL_TRAIN
N_REPLICAS   = 5
SEEDS        = [42, 123, 456, 789, 1024]

synthetic_registry = []   # list of dicts for structured bookkeeping

for synth_name, synth in fitted.items():
    for prop in PROPORTIONS:
        n_synth = int(N_REAL_TRAIN * prop)
        for replica_idx, seed in enumerate(SEEDS):
            import random
            np.random.seed(seed)
            random.seed(seed)
            df_s = synth.sample(num_rows=n_synth)
            df_s["data_origin"] = "synthetic"
            df_s["synth_model"] = synth_name
            df_s["synth_proportion"] = prop
            df_s["synth_replica"] = replica_idx
            df_s["synth_seed"] = seed

            synthetic_registry.append({
                "model": synth_name,
                "proportion": prop,
                "replica": replica_idx,
                "seed": seed,
                "n_rows": len(df_s),
                "df": df_s,
            })
```

> **On reproducibility with SDV ≥ 1.0.** The synthesiser's internal RNG is initialised at `.fit()` time. To obtain reproducible samples across runs, set `np.random.seed()` and `random.seed()` immediately before each `.sample()` call. The notebook documents the exact seed sequence used.

### 4.4 Biological Validity Filter

Before any synthetic record can be used in a downstream experiment, it must pass the same biological validity checks defined in Part 1 (§4.5). This is non-negotiable: feeding biologically impossible fish to a predictive model trains that model on noise.

```python
def biological_validity_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply morphometric validity constraints for Solea senegalensis.
    Returns the input dataframe with a boolean column 'bio_valid'.

    Constraints:
      - All measurements strictly positive
      - K_area = weight_g / (length_cm × width_cm) ∈ [0.08, 0.50]  (Froese 2006)
      - width / length ∈ [0.20, 0.55]  (bilateral symmetry constraint)
      - thickness / length ∈ [0.02, 0.12]  (flatfish morphology)
    """
    df = df.copy()

    positive = (
        (df["weight_g"] > 0) &
        (df["length_cm"] > 0) &
        (df["width_cm"] > 0) &
        (df["thickness_cm"] > 0)
    )

    k_area   = df["weight_g"] / (df["length_cm"] * df["width_cm"])
    wl_ratio = df["width_cm"] / df["length_cm"]
    tl_ratio = df["thickness_cm"] / df["length_cm"]

    df["bio_valid"] = (
        positive &
        k_area.between(0.08, 0.50) &
        wl_ratio.between(0.20, 0.55) &
        tl_ratio.between(0.02, 0.12)
    )
    return df
```

The filter is applied independently to each replica. Records that fail are retained in the full registry (for diagnostic purposes) but **excluded from any training set** used in downstream predictive experiments. We record the pass rate per model and replica — a consistently low pass rate is a diagnostic signal that the synthesiser is struggling with the morphometric constraints of *Solea senegalensis*.

### 4.5 Saving Synthetic Datasets

All synthetic replicas are persisted to `data/synthetic/` in Parquet format. The filename encodes the synthesiser, proportion and replica index — making every file self-describing without needing to parse a separate index.

```python
from pathlib import Path
import json

SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

manifest = []

for entry in synthetic_registry:
    df_valid = entry["df"][entry["df"]["bio_valid"]].copy()
    fname = (
        f"synthetic_{entry['model']}_"
        f"p{int(entry['proportion']*100):03d}_"
        f"r{entry['replica']:02d}.parquet"
    )
    fpath = SYNTHETIC_DIR / fname
    df_valid.to_parquet(fpath, index=False)

    manifest.append({
        "model": entry["model"],
        "proportion": entry["proportion"],
        "replica": entry["replica"],
        "seed": entry["seed"],
        "n_generated": len(entry["df"]),
        "n_valid": int(entry["df"]["bio_valid"].sum()),
        "pass_rate": float(entry["pass_rate"]),
        "path": str(fpath),
    })

with open(SYNTHETIC_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved {len(manifest)} synthetic datasets to {SYNTHETIC_DIR}")
```

The `manifest.json` file serves as the registry for all downstream experiments — Part 3 loads it to enumerate available datasets without scanning the directory.

---

## 5. Did the Synthesisers Learn Anything Useful?

Before evaluating predictive utility, we need to ask a more basic question: are the synthetic datasets *statistically faithful* to the training data, and do they produce *biologically credible* fish? A synthesiser that fails these checks is not worth evaluating further as a data augmentation tool.

We structure the evaluation around four dimensions, in increasing order of difficulty.

### 5.1 Univariate Fidelity

The simplest check: does each variable's marginal distribution in the synthetic data resemble the corresponding distribution in real training data?

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FEATURES_ALL = ["weight_g", "length_cm", "width_cm", "thickness_cm"]

def univariate_fidelity_report(df_real, df_synth, synth_name):
    """
    For each variable: compare moments, KS statistic, and plot
    overlapping KDE (real vs synthetic).
    """
    report = {}
    fig, axes = plt.subplots(1, len(FEATURES_ALL), figsize=(16, 4))

    for ax, col in zip(axes, FEATURES_ALL):
        real_vals  = df_real[col].dropna().values
        synth_vals = df_synth[col].dropna().values

        ks_stat, ks_p = stats.ks_2samp(real_vals, synth_vals)
        report[col] = {
            "real_mean":  real_vals.mean(),
            "synth_mean": synth_vals.mean(),
            "real_std":   real_vals.std(),
            "synth_std":  synth_vals.std(),
            "ks_stat":    ks_stat,
            "ks_p":       ks_p,
        }

        sns.kdeplot(real_vals,  ax=ax, label="Real train", color="#2166ac", fill=True, alpha=0.3)
        sns.kdeplot(synth_vals, ax=ax, label=synth_name,   color="#d73027", fill=True, alpha=0.3)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)

    plt.suptitle(f"Marginal distributions — {synth_name} vs Real", y=1.02)
    plt.tight_layout()
    return report, fig
```

Run once per synthesiser on a single representative replica (100% proportion, replica 0), this gives a first, illustrative look — three KDE panels per model, four variables each:

*Figure 5.1a — to be inserted: `results/figures/fig_univariate_<synth_name>.png` for each synthesiser (real vs. single-replica synthetic KDE, generated by `univariate_fidelity_report`).*

**Interpreting Kolmogorov-Smirnov (KS) statistics with caution.** The two-sample KS test measures whether two sets of observations are likely to have been drawn from the same continuous distribution. It does so non-parametrically: it computes the maximum vertical distance between the empirical cumulative distribution functions of the two samples and asks how often a gap that large would arise by chance if both samples shared the same underlying distribution. Here, we use it to detect systematic differences between the marginal distribution of each variable in the real training data and in the synthetic data. With n = 167 real records and a similar number of synthetic samples, however, the test has moderate power. A non-significant p-value does not guarantee distributional equivalence — it may simply reflect limited statistical power. We therefore treat KS results as one indicator among many, and prioritise visual inspection of overlapping KDEs.

**Why one replica is not enough.** All three synthesisers sample stochastically, and CTGAN's adversarial training is expected (§2.2) to be the most seed-sensitive of the three. A KS statistic and a KDE plot built from a single arbitrary replica cannot distinguish a systematic distortion of the generator from ordinary sampling noise in that one draw — and, as the retracted `weight_g` finding for CTGAN below shows, the two are not always easy to tell apart by eye. The check below repeats the same comparison two ways across all 5 replicas SDV can sample from each fitted generator: **pooled** (every valid replica's rows concatenated into one larger synthetic sample, for a single higher-power KS test) and **per-replica** (KS run separately on each of the 5 seeds, to see whether the result is stable or swings from replica to replica).

```python
from typing import Sequence

def univariate_fidelity_replicated(
    df_real: pd.DataFrame,
    registry: list[dict],
    synth_name: str,
    proportion: float = 1.00,
    features: Sequence[str] = FEATURES_ALL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute univariate KS fidelity for one synthesiser at one augmentation
    proportion, aggregated across all its replicas. Returns (df_pooled,
    df_per_replica): pooled KS/Δmean on the concatenation of all valid
    replicas, and KS/Δmean computed separately on each replica.
    """
    entries = [e for e in registry if e["model"] == synth_name and e["proportion"] == proportion]
    real_vals_by_col = {col: df_real[col].dropna().values for col in features}

    per_replica_records, pooled_synth_vals = [], {col: [] for col in features}
    for entry in entries:
        df_valid = entry["df"][entry["df"]["bio_valid"]]
        if len(df_valid) < 5:
            continue
        for col in features:
            s_vals, r_vals = df_valid[col].dropna().values, real_vals_by_col[col]
            ks_stat, ks_p = stats.ks_2samp(r_vals, s_vals)
            per_replica_records.append({
                "model": synth_name, "replica": entry["replica"], "seed": entry["seed"],
                "variable": col, "n_synth": len(s_vals),
                "ks_stat": ks_stat, "ks_p": ks_p, "delta_mean": s_vals.mean() - r_vals.mean(),
            })
            pooled_synth_vals[col].append(s_vals)

    df_per_replica = pd.DataFrame(per_replica_records)

    pooled_records = []
    for col in features:
        s_pooled, r_vals = np.concatenate(pooled_synth_vals[col]), real_vals_by_col[col]
        ks_stat, ks_p = stats.ks_2samp(r_vals, s_pooled)
        pooled_records.append({
            "model": synth_name, "variable": col, "n_synth_pooled": len(s_pooled),
            "ks_stat": ks_stat, "ks_p": ks_p, "delta_mean": s_pooled.mean() - r_vals.mean(),
        })
    return pd.DataFrame(pooled_records), df_per_replica


# ── Run for every synthesiser, same 100% reference point as above ──────────
pooled_all, per_replica_all = [], []
for synth_name in fitted.keys():
    df_pooled, df_per_rep = univariate_fidelity_replicated(df_train, synthetic_registry, synth_name)
    pooled_all.append(df_pooled)
    per_replica_all.append(df_per_rep)

df_pooled_all = pd.concat(pooled_all, ignore_index=True)
df_per_replica_all = pd.concat(per_replica_all, ignore_index=True)

# Stability diagnostic: how much does KS/Δmean vary across the 5 seeds?
df_stability = (
    df_per_replica_all.groupby(["model", "variable"])
    .agg(ks_mean=("ks_stat", "mean"), ks_std=("ks_stat", "std"),
         delta_mean_mean=("delta_mean", "mean"), delta_mean_std=("delta_mean", "std"),
         frac_significant=("ks_p", lambda s: float((s < 0.05).mean())))
    .round(4)
)
```

*Figure 5.1b — to be inserted: `results/figures/fig_univariate_pooled_<synth_name>.png` for each synthesiser — real KDE (bold blue) vs. each of the 5 replica KDEs (thin, semi-transparent red) vs. the pooled KDE across all valid replicas (bold dashed red), generated by a companion `plot_univariate_fidelity_pooled` function. A tight bundle of thin red lines means a stable result (TVAE, most of GaussianCopula); a wide spread means the pooled/single-replica number should not be trusted at face value (CTGAN's `weight_g`, below).*

**Reading the results.** The tables below are the replica-aggregated check, not a single arbitrary draw: Table 5.1a is the pooled KS statistic on every valid replica's synthetic sample concatenated together; Table 5.1b is the per-replica stability check — how many of the 5 seeds reach significance individually. These numbers come from a full re-run of the notebook, refitting all three synthesisers from scratch under the same nominal seed, rather than the original pass — and the comparison across runs turns out to matter as much as the comparison across replicas within a run. GaussianCopula's results are essentially unchanged; TVAE shifts moderately; CTGAN's headline finding from the original pass — the `length_cm` distortion — does not survive the re-run at all. That comparison is discussed in full after the tables, and it changes the conclusion of this subsection.

*Table 5.1a — Pooled KS (primary fidelity estimate; all valid replicas at the 100% proportion concatenated per model)*

| Model | Variable | n synthetic (pooled) | KS | p | Δmean |
|---|---|---|---|---|---|
| TVAE | weight_g | 822 | 0.053 | 0.810 | −0.201 |
| TVAE | length_cm | 822 | 0.059 | 0.697 | −0.080 |
| TVAE | width_cm | 822 | 0.052 | 0.826 | −0.026 |
| TVAE | thickness_cm | 822 | 0.032 | 0.998 | +0.003 |
| GaussianCopula | weight_g | 764 | 0.188 | < 0.001 | +0.460 |
| GaussianCopula | length_cm | 764 | 0.105 | 0.090 | +0.140 |
| GaussianCopula | width_cm | 764 | 0.161 | 0.001 | +0.075 |
| GaussianCopula | thickness_cm | 764 | 0.234 | < 0.0001 | +0.012 |
| CTGAN | weight_g | 355 | 0.111 | 0.109 | −0.282 |
| CTGAN | length_cm | 355 | 0.101 | 0.181 | +0.131 |
| CTGAN | width_cm | 355 | 0.104 | 0.158 | −0.170 |
| CTGAN | thickness_cm | 355 | 0.184 | 0.001 | −0.020 |

*Table 5.1b — Per-replica stability (mean ± std of KS and Δmean across the 5 seeds; last column = how many of the 5 individual replica-level tests reach p < 0.05)*

| Model | Variable | KS (mean ± std) | Δmean (mean ± std) | Replicas significant |
|---|---|---|---|---|
| TVAE | weight_g | 0.076 ± 0.022 | −0.201 ± 0.253 | 0/5 |
| TVAE | length_cm | 0.077 ± 0.031 | −0.081 ± 0.127 | 0/5 |
| TVAE | width_cm | 0.072 ± 0.017 | −0.027 ± 0.052 | 0/5 |
| TVAE | thickness_cm | 0.048 ± 0.018 | +0.003 ± 0.010 | 0/5 |
| GaussianCopula | weight_g | 0.196 ± 0.025 | 0.462 ± 0.240 | 5/5 |
| GaussianCopula | length_cm | 0.119 ± 0.029 | 0.141 ± 0.123 | 1/5 |
| GaussianCopula | width_cm | 0.171 ± 0.034 | 0.075 ± 0.054 | 4/5 |
| GaussianCopula | thickness_cm | 0.235 ± 0.035 | 0.012 ± 0.013 | 5/5 |
| CTGAN | weight_g | 0.124 ± 0.009 | −0.285 ± 0.127 | 0/5 |
| CTGAN | length_cm | 0.121 ± 0.047 | 0.131 ± 0.080 | 0/5 |
| CTGAN | width_cm | 0.146 ± 0.025 | −0.169 ± 0.048 | 0/5 |
| CTGAN | thickness_cm | 0.183 ± 0.025 | −0.021 ± 0.012 | 2/5 |

TVAE's clean pass is confirmed rather than merely suggested: pooled KS stays below 0.06 on every variable (p ≥ 0.69), and none of its 20 individual replica-level tests (4 variables × 5 seeds) reaches significance. This is the most stable result in the table.

GaussianCopula's three-variable pattern holds up, and more strongly than a single replica suggested. `weight_g` (pooled KS = 0.188, p < 0.001) and `thickness_cm` (pooled KS = 0.234, p < 0.001) are significant in all 5 of their 5 replicas; `width_cm` (pooled KS = 0.161, p = 0.001) in 4 of 5. `length_cm` stays non-significant (pooled p = 0.090; only 1 of 5 replicas significant). The location-vs-shape distinction is confirmed too: `thickness_cm` carries the largest KS statistic in the table (0.234) together with the smallest pooled Δmean (+0.012, tightly reproduced at 0.012 ± 0.013 across replicas) — a stable shape mismatch, not a location bias, consistent with the `default_distribution="norm"` constraint set in §4.2.

CTGAN is where the *re-run* changes the story, not the aggregation. None of its four variables is consistently significant across replicas: `weight_g` (pooled KS = 0.111, p = 0.109; per-replica KS = 0.124 ± 0.009, Δmean = −0.285 ± 0.127) and `length_cm` (pooled KS = 0.101, p = 0.181; per-replica KS = 0.121 ± 0.047, Δmean = 0.131 ± 0.080) are both non-significant, 0 of 5 replicas each. `width_cm` is likewise non-significant (pooled KS = 0.104, p = 0.158, 0/5). Only `thickness_cm` shows a stable, if partial, deviation: pooled KS = 0.184 (p = 0.001), 2 of 5 replicas individually significant. This is a striking reversal of the previous pass, where `length_cm` was the single most robust finding in the table (5/5 replicas significant, pooled KS = 0.275) — and it did not happen because the earlier finding was a single-replica artefact; it happened on a full re-fit with the same nominal seed. The full implication of that is discussed below, after this paragraph.

One further result falls out of this analysis that the single-replica check could not surface. The pooled sample sizes differ sharply across synthesisers at the same nominal 100% proportion — 822 of 835 generated rows valid for TVAE, 764 of 835 for GaussianCopula, but only 355 of 835 for CTGAN. That is consistent with the full biological-validity pass rates reported in §5.3 (TVAE 98.4%, GaussianCopula 91.0%, CTGAN 39.1%, aggregated across all proportions and replicas): CTGAN's fidelity numbers above are computed on the smallest, most power-constrained sample of the three, for the model whose failures most need scrutiny.

One qualification from the original draft has now been tested directly, rather than just flagged, and the result is not reassuring. Every replica in the tables above resamples the *same* fitted generator — but the tables themselves come from a full re-run of this notebook, which refit every synthesiser from scratch under the same nominal `RANDOM_STATE=42` (§4.2). That re-fit is exactly the training-stability test the original draft could only pose as an open question. GaussianCopula's numbers are reproduced almost to the decimal place across the re-run — expected, since it has no stochastic optimiser to reseed. CTGAN's `length_cm` finding does not reproduce on this re-fit at all, and TVAE shifts by a smaller but still real margin on `weight_g` and `length_cm` (Table 5.1a). The most likely explanation — not yet confirmed, and worth treating as a hypothesis rather than an established fact — is that `RANDOM_STATE=42`, together with the `np.random.seed()`/`random.seed()` calls in §4.3, only fixes NumPy's and Python's global RNGs, which govern *sampling* from an already-fitted generator; PyTorch's own RNG, which governs *training* CTGAN's and TVAE's networks inside SDV, is left unseeded by the current setup. If that is right, the univariate fidelity numbers reported for CTGAN and, to a lesser extent, TVAE describe one particular training run rather than a stable property of the model class, and this notebook is not reproducible end-to-end under its current seeding. That reclassifies this from an open question into a concrete limitation: §13 should seed every RNG a library uses internally, not only the ones it exposes, and §12's sensitivity analysis should treat training-seed variability as a primary axis rather than a follow-up layered on top of sampling-seed variability alone.

### 5.2 Multivariate Fidelity — Allometric Preservation

Marginal fidelity is necessary but not sufficient. The biological content of this dataset resides in the *joint* structure: the allometric relationship that links weight to the three morphometric dimensions. A synthesiser that reproduces each variable's marginal independently, but destroys the correlation structure, would generate fish with incoherent morphologies — large lengths paired with tiny widths, for instance.

We measure multivariate fidelity through two lenses.

**Correlation matrix comparison:**

```python
def correlation_delta_heatmap(df_real, df_synth, synth_name, method="pearson"):
    """
    Compute correlation matrices for real and synthetic data,
    then plot the element-wise difference.
    """
    corr_real  = df_real[FEATURES_ALL].corr(method=method)
    corr_synth = df_synth[FEATURES_ALL].corr(method=method)
    delta = corr_synth - corr_real

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kwargs = dict(vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="coolwarm", square=True)

    sns.heatmap(corr_real,  ax=axes[0], **kwargs, cbar=False)
    axes[0].set_title("Real train")

    sns.heatmap(corr_synth, ax=axes[1], **kwargs, cbar=False)
    axes[1].set_title(f"Synthetic — {synth_name}")

    sns.heatmap(delta,      ax=axes[2], **kwargs)
    axes[2].set_title("Δ (Synthetic − Real)")

    plt.suptitle(f"{method.capitalize()} correlation comparison — {synth_name}", y=1.02)
    plt.tight_layout()
    return delta
```

Run on a single representative replica, this is a reasonable first look — but §5.1 already showed that a single replica of a stochastic generator (CTGAN especially) can mistake sampling noise for a systematic property, and there is no reason the correlation structure would be immune to the same risk. The fix is the same one applied there: pool every valid replica's rows into one larger synthetic sample for the primary Δ estimate, and compute the element-wise standard deviation of the delta matrix across the 5 individual replicas as a stability diagnostic.

```python
def correlation_delta_replicated(df_real, registry, synth_name, proportion=1.00,
                                  features=FEATURES_ALL, method="pearson"):
    """
    Pearson/Spearman correlation fidelity for one synthesiser, aggregated
    across all its replicas at one proportion. Returns the real correlation
    matrix, the pooled Δ (synthetic_pooled − real), the element-wise Std(Δ)
    across replicas, and the list of per-replica Δ matrices.
    """
    entries = [e for e in registry if e["model"] == synth_name and e["proportion"] == proportion]
    corr_real = df_real[features].corr(method=method)

    valid_dfs = [e["df"][e["df"]["bio_valid"]] for e in entries]
    valid_dfs = [d for d in valid_dfs if len(d) >= 5]

    df_pooled = pd.concat(valid_dfs, ignore_index=True)
    delta_pooled = df_pooled[features].corr(method=method) - corr_real

    per_replica_deltas = [d[features].corr(method=method) - corr_real for d in valid_dfs]
    delta_std = pd.DataFrame(
        np.stack([d.values for d in per_replica_deltas]).std(axis=0),
        index=features, columns=features,
    )
    return corr_real, delta_pooled, delta_std, per_replica_deltas


# ── Run for every synthesiser; summarise as one row each ───────────────────
corr_summary_records = []
for synth_name in fitted.keys():
    _, delta_pooled, delta_std, _ = correlation_delta_replicated(
        df_train, synthetic_registry, synth_name, proportion=1.00,
    )
    off_diag = ~np.eye(len(FEATURES_ALL), dtype=bool)
    corr_summary_records.append({
        "model": synth_name,
        "max_abs_delta_pooled": float(np.abs(delta_pooled.values[off_diag]).max()),
        "max_std_delta_replicas": float(delta_std.values[off_diag].max()),
    })

df_corr_summary = pd.DataFrame(corr_summary_records)
```

*Figure 5.2a — to be inserted: `results/figures/fig_corr_pooled_<synth_name>.png` for each synthesiser — four panels: real correlation matrix, pooled synthetic correlation matrix, Δ pooled (synthetic − real), and Std(Δ) across the 5 replicas. A near-zero, uniformly dark "Std(Δ)" panel means the correlation-fidelity read is stable; visible hot spots there mean the corresponding Δ cell should not be trusted from a single replica alone — exactly the caution already established for the univariate results.*

*Table 5.2a — Correlation fidelity summary, off-diagonal pairs only*

| Model | max &#124;Δ pooled&#124; | max Std(Δ) across replicas |
|---|---|---|
| TVAE | 0.159 | 0.026 |
| GaussianCopula | 0.032 | 0.035 |
| CTGAN | 0.757 | 0.100 |

CTGAN's correlation structure is still severely broken, though less dramatically than in the previous pass: a pooled Δ of 0.757 on a statistic bounded to [-1, 1] means at least one pairwise correlation that real fish preserve tightly is substantially eroded, or possibly inverted, in the synthetic data. This is not a single-replica artefact: the per-replica std for that same cell is only 0.100, more than seven times smaller than the deviation itself, so the distortion is consistent across all 5 seeds. This is a different picture from §5.1, where CTGAN's marginals — including `length_cm`, previously its clearest univariate failure — did not reproduce as distorted on this re-run: CTGAN's problem here is concentrated in the *joint* structure between variables, not the marginals themselves, and it is corroborated by the collinearity diagnostic later in this section (Table 5.2c) and by CTGAN's still-low biological-validity pass rate (39.1%, §5.3).

GaussianCopula has the smallest pooled correlation deviation of the three (0.032) — perhaps unsurprisingly, since a Gaussian copula is explicitly a model *of* the dependence structure, so preserving correlations is closer to what it is built to do, even though §5.1 already showed it distorts three of the four marginal shapes.

TVAE is the more interesting result. It had the cleanest univariate fidelity in §5.1 — no significant KS deviation on any variable, any replica — yet its correlation fidelity here (0.159) is nearly five times larger than GaussianCopula's (0.032). Its per-replica stability is good in absolute terms and, in fact, tighter than GaussianCopula's (max Std(Δ) = 0.026 vs. 0.035) — so the larger deviation is a consistent property of TVAE's fit, not sampling noise around a value that happens to look large. This is precisely the failure mode the opening paragraph of this section warned about: a synthesiser can reproduce every marginal distribution correctly while still generating fish whose *joint* morphometric structure is off. Univariate fidelity and multivariate fidelity are answering different questions, and this dataset's clearest evidence that they can disagree is TVAE, not CTGAN — whose §5.1 marginal problems, on this re-run, did not hold up at all.

**Allometric coefficient stability:**

We refit the multivariate log-log model W = a·L^b₁·A^b₂·E^b₃ separately on real training data and on each synthetic replica, then compare the estimated exponents.

The original version of this check pooled all four augmentation proportions (25–200%) together with all 5 replicas into a single `.groupby("source").describe()`. That mixing is a problem, not just a simplification: the number of rows generated per replica scales with the proportion (~42 at 25% vs. ~334 at 200%, before the biological filter), so the OLS exponent estimates at different proportions have different sampling variance by construction — pooling them inflates the reported spread with an effect that has nothing to do with the generator's stability, and hides whether exponent bias depends on how much synthetic data is used, which is a question for Part 3's dose-response design, not for this fidelity check. The version below fixes the proportion at 100% — the same reference point used throughout §5.1–§5.3 — and reports exponent stability only across the 5 replicas at that one proportion. It also records, rather than silently drops, any replica excluded by the existing `n_valid < 20` filter (relevant mainly for CTGAN: ~42 generated rows × ~45% biological pass rate, §5.3, sits close to that threshold at low proportions — though not at the 100% proportion used here).

```python
from sklearn.linear_model import LinearRegression

def fit_allometric_loglog(df, features=["length_cm", "width_cm", "thickness_cm"],
                           target="weight_g"):
    """Fit log(W) = log(a) + b1*log(L) + b2*log(A) + b3*log(E) via OLS."""
    X = np.log(df[features].values)
    y = np.log(df[target].values)
    model = LinearRegression().fit(X, y)
    coefs = dict(zip(["log_a"] + [f"b_{f}" for f in features],
                     [model.intercept_] + list(model.coef_)))
    coefs["a"] = np.exp(model.intercept_)
    coefs["R2_loglog"] = model.score(X, y)
    return coefs


def allometric_coefs_replicated(df_real, registry, proportion=1.00,
                                 features=FEATURES, target=TARGET, min_valid_rows=20):
    """
    Refit the allometric model per replica of each synthesiser at a FIXED
    proportion (default 100%), and summarise exponent stability across
    replicas. Replicas below `min_valid_rows` valid rows are excluded and
    counted explicitly rather than dropped silently.
    """
    ref_coefs = fit_allometric_loglog(df_real, features=features, target=target)

    records, excluded = [], []
    for entry in registry:
        if entry["proportion"] != proportion:
            continue
        df_valid = entry["df"][entry["df"]["bio_valid"]]
        if len(df_valid) < min_valid_rows:
            excluded.append({"model": entry["model"], "replica": entry["replica"], "n_valid": len(df_valid)})
            continue
        coefs = fit_allometric_loglog(df_valid, features=features, target=target)
        records.append({"model": entry["model"], "replica": entry["replica"], "n_valid": len(df_valid), **coefs})

    df_coefs = pd.DataFrame(records)
    df_excluded = pd.DataFrame(excluded)
    b_cols = [f"b_{f}" for f in features]
    df_summary = df_coefs.groupby("model")[b_cols].agg(["mean", "std", "count"]).round(4)
    return ref_coefs, df_coefs, df_summary, df_excluded


ref_coefs, df_coefs, df_allometric_summary, df_excluded_replicas = allometric_coefs_replicated(
    df_real=df_train, registry=synthetic_registry, proportion=1.00,
)
```

*Figure 5.2b — to be inserted: `results/figures/fig_allometric_stability.png` — boxplot of b₁ (length), b₂ (width), b₃ (thickness) per synthesiser across its 5 replicas at the 100% proportion, with a dashed line at the real training value for each exponent. (The notebook's existing §10b cell currently pools all four proportions into this plot too; re-run it filtered to `proportion == 1.00` for consistency with Table 5.2b below, and keep the all-proportions version, if useful, as a separate exploratory figure for Part 3.)*

*Table 5.2b — Allometric exponent stability at the 100% proportion, mean ± std across 5 replicas (real reference: b₁ ≈ 1.621, b₂ ≈ 0.876, b₃ ≈ 0.393)*

| Model | b₁ (length) | b₂ (width) | b₃ (thickness) | Replicas used | Replicas excluded (n<20) |
|---|---|---|---|---|---|
| TVAE | 0.917 ± 0.217 | 0.817 ± 0.091 | 0.780 ± 0.101 | 5 | 0 |
| GaussianCopula | 0.624 ± 0.279 | 1.922 ± 0.248 | 0.319 ± 0.323 | 5 | 0 |
| CTGAN | 0.752 ± 0.119 | 0.731 ± 0.171 | 0.113 ± 0.325 | 5 | 0 |

**Reading the results.** No synthesiser reproduces the real allometric structure well. All three exponents are systematically biased for all three models. Unlike §5.1's univariate picture on this re-run — a largely clean pair (TVAE, CTGAN) against one consistent failure (GaussianCopula) — the allometric exponents here show three distinct failure modes, with no model doing well across the board.

The cross-check with §5.1 that the previous pass reported for b₁ (length) does not survive this re-run, and the retraction is informative in its own right. Ranked by how far each model's mean falls from the real value (1.621): TVAE (0.917, off by 0.70) < CTGAN (0.752, off by 0.87) < GaussianCopula (0.624, off by 1.00). On this re-run, §5.1's `length_cm` marginal fidelity ranks TVAE and CTGAN as similarly good (0 of 5 replicas significant for both) and GaussianCopula as the weakest (1/5 significant) — yet CTGAN's b₁ is nearly as biased as GaussianCopula's, despite CTGAN's marginal being, by the KS test, indistinguishable from TVAE's. A well-reproduced marginal distribution for `length_cm` does not guarantee a well-estimated allometric exponent for it: the regression coefficient depends on the *joint* relationship between `length_cm` and the other two predictors, not on `length_cm`'s marginal shape alone — consistent with §5.2's correlation and VIF findings for CTGAN below (a broken joint structure), and exactly the kind of dependency the four-dimension evaluation design in §2.4 was meant to be able to detect. The earlier pass's claim that a badly-reproduced marginal feeding a regression on that same variable produces a badly biased coefficient "as the expected, coherent result, not a coincidence" was itself an artefact of one training run in which CTGAN happened to fail on both dimensions at once; this re-run shows the two are not, in general, coupled that tightly.

GaussianCopula's b₂ (width) is the single most distorted number in this table — 1.922 against a real value of 0.876, more than double — and reproduced with only moderate consistency (std = 0.248), in fact the largest per-exponent spread of the three models on b₂ (TVAE: 0.091; CTGAN: 0.171), not the tightest as an earlier pass of this analysis described it. That is a striking result next to Table 5.2a, where GaussianCopula had the *smallest* pooled correlation deviation of the three synthesisers (0.032). Those two facts are not necessarily in tension: `length_cm`, `width_cm` and `thickness_cm` are highly correlated with each other in real fish (a growing flatfish scales in most dimensions together), and a multivariate OLS fit on near-collinear predictors is known to have unstable individual-coefficient estimates even when the overall correlation structure — and the fit's R² — barely move. Under that reading, GaussianCopula's small residual correlation error could still be large enough, concentrated in the *relative* correlations among the three predictors rather than their absolute magnitude, to swing how the regression apportions credit between `b_width_cm` and the other two exponents. This is a hypothesis, not an established conclusion here: confirming it needs the log-log fit's R² (already computed by `fit_allometric_loglog` but dropped from this summary table) and a collinearity diagnostic — variance inflation factors or the condition number of the `[log L, log W, log T]` design matrix, for real and pooled-synthetic data alike. If R² stays close to the real-data value (0.982) despite the coefficient swing, that would support a multicollinearity artefact in how the exponents are individually estimated rather than a genuine loss of predictive information; if R² also drops substantially, the distortion is real and not just a coefficient-attribution problem.

CTGAN's `b₃` (thickness) is severely biased and highly unstable — 0.113 ± 0.325 against a real value of 0.393 — with a standard deviation nearly three times the mean itself, the least reproducible exponent-model pair in the table. This is a different failure mode from a clean collapse to zero: rather than consistently finding no thickness–weight relationship, individual replicas swing between strongly negative and strongly positive estimates for this exponent. Combined with the correlation breakdown in Table 5.2a and the 39.1% biological-validity pass rate in §5.3, this is still a case of CTGAN failing on multiple independent checks, but the mechanism reads differently on this re-run — not a generator that has settled on a wrong-but-stable relationship, but one whose estimate of this particular exponent is essentially unconstrained by the training data.

TVAE's `b₃` shows the opposite problem to CTGAN's: not volatility, but a precise, stable overestimate (0.780 ± 0.101 against a real 0.393 — essentially double, and one of the tighter stds in the table, though not the tightest — TVAE's own `b₂` is more stable still at 0.091). A tightly reproduced *wrong* answer is arguably a more concerning failure mode than a noisy one, because it would not be caught by looking at variance or stability alone — TVAE's `thickness_cm` marginal passed every check in §5.1, and its correlation fidelity in §5.2, while worse than GaussianCopula's, was itself fairly stable (Table 5.2a), yet the specific three-way relationship this exponent encodes is confidently miscalibrated.

The practical implication carries directly into §5.4: TSTR trains a predictive model on exactly this log-log allometric relationship, using synthetic data. If none of the three synthesisers preserve it, the TSTR benchmark should be expected, before even running it, to underperform TRTR for all three — the open question is by how much, and whether that gap is small enough at some augmentation proportion to still be useful in combination with real data, which is exactly what Part 3 is designed to test.

**Testing the multicollinearity hypothesis.** The paragraph above proposed a specific, checkable explanation for GaussianCopula's `b_width_cm` swing: three highly collinear predictors can make individual OLS coefficients unstable even when the fit itself, and the overall correlation structure, barely change. Two diagnostics decide this — the log-log fit's R² (already computed by `fit_allometric_loglog`, but dropped from Table 5.2b), and the variance inflation factor (VIF) of each log-transformed predictor, computed here without adding a new dependency:

```python
from sklearn.linear_model import LinearRegression

def compute_vif(X, feature_names):
    """VIF_i = 1 / (1 - R2_i), R2_i from regressing predictor i on the rest."""
    vif = {}
    for i, feat in enumerate(feature_names):
        X_others = np.delete(X, i, axis=1)
        r2_i = LinearRegression().fit(X_others, X[:, i]).score(X_others, X[:, i])
        vif[feat] = float(1.0 / (1.0 - r2_i)) if r2_i < 1.0 else float("inf")
    return vif

# Real reference + pooled synthetic data (100% proportion) per model
X_real = np.log(df_train[FEATURES].values)
vif_real = compute_vif(X_real, FEATURES)
# ... repeated for each synthesiser's pooled valid rows; R² taken from the
# per-replica fits already computed in allometric_coefs_replicated (§10c)
```

*Table 5.2c — Log-log fit R² (mean ± std across 5 replicas at 100% proportion) and predictor VIF (computed on pooled valid rows; real training data has no replicate variability)*

| Source | R²_loglog | VIF length | VIF width | VIF thickness |
|---|---|---|---|---|
| Real train | 0.982 (single fit) | 10.16 | 8.80 | 4.19 |
| TVAE | 0.766 ± 0.019 | 3.39 | 3.28 | 3.50 |
| GaussianCopula | 0.867 ± 0.036 | 8.10 | 6.85 | 3.54 |
| CTGAN | 0.361 ± 0.086 | 1.29 | 1.27 | 1.03 |

The hypothesis is confirmed only in part. GaussianCopula's R² does drop from the real value (0.982 → 0.867) — this is not a case of the fit being untouched while one coefficient silently absorbs all the change, so the coefficient-attribution story is not the whole explanation. But the drop is far smaller than for the other two synthesisers, and its VIF pattern (8.10 / 6.85 / 3.54) stays closest to real data's own collinearity structure (10.16 / 8.80 / 4.19) — same declining order, same regime of moderate-to-severe collinearity. That is consistent with the original intuition in a weaker form: GaussianCopula preserves *most* of the real predictive relationship and *most* of the real collinearity structure, and what little structure it does lose is enough, under near-collinear predictors, to swing `b_width_cm` further than the modest R² drop alone would suggest.

The bigger finding is CTGAN's collinearity, and on this re-run it sharpens rather than changes the overall reading of this section. Its VIF collapses to essentially 1.0 on all three predictors (1.29 / 1.27 / 1.03) — length, width and thickness become almost uncorrelated with each other in CTGAN's synthetic fish, against a real training population where they are, by construction of how a flatfish grows, strongly collinear (VIF 4–10). That is not a subtle distortion; it is the removal of the co-scaling relationship that the entire allometric model depends on, and it explains, mechanistically, why CTGAN's R² collapses to 0.361 — roughly a third of the variance a model should explain, against 98% for real data. What this re-run clarifies, against the previous pass, is exactly *which* CTGAN symptoms this one mechanism explains and which it does not. It accounts for the broken correlation matrix (Table 5.2a, Δ = 0.757), the unstable `b_thickness_cm` estimate (Table 5.2b), and the collapsed in-sample R² above — three findings that are all, directly, about the joint relationship between the three predictors, which is precisely what a collinearity collapse destroys. It does *not* explain the univariate marginal fidelity of §5.1, which on this re-run is largely clean for CTGAN (only `thickness_cm` shows a partial, stable deviation) — a synthesiser can decouple three variables from each other while still sampling each one individually from close to the right distribution. The 39.1% biological-validity pass rate (§5.3) sits closer to the joint-structure failures than the marginal ones, since the validity checks are themselves ratios and products of the three dimensions (§4.4) — consistent with a generator that gets each dimension roughly right in isolation but not in combination.

TVAE's numbers are the one result here that complicates, rather than confirms, the picture built up so far. Its R² (0.766) is lower than GaussianCopula's (0.867), despite TVAE having the cleanest univariate marginals in §5.1 and a smaller correlation Δ than CTGAN in Table 5.2a — reinforcing that no single fidelity dimension predicts another, which is exactly why §2.4 insisted on checking all four independently rather than picking a winner early. Its VIF pattern is also distinctive: a narrow, non-monotonic spread across the three predictors (3.39 / 3.28 / 3.50), where both real data and GaussianCopula show a clear declining gradient (length > width > thickness). That TVAE's synthetic fish homogenise collinearity rather than simply weakening it, proportionally, across all three dimensions — and do so in a way that does not even preserve the real data's ordering among the three — is an observation worth carrying into Part 3, not yet an explained one; it would need a targeted follow-up (e.g. comparing conditional variance structure in TVAE's latent space) to say anything stronger than that the pattern exists.

### 5.3 Biological Validity Summary

The pass rates computed in §4.4 are aggregated here into a diagnostic table, across all 4 augmentation proportions and 5 replicas per synthesiser (20 runs each):

| Synthesiser | Mean pass rate | Std | Min | Max |
|---|---|---|---|---|
| GaussianCopula | 91.0% | 2.9% | 85.4% | 95.2% |
| CTGAN | 39.1% | 6.4% | 26.5% | 51.2% |
| TVAE | 98.4% | 1.2% | 95.1% | 100.0% |

Pass rates below 80% warrant investigation: the synthesiser may be generating morphometric combinations that violate the geometric constraints of *Solea senegalensis*. Pass rates consistently above 95% — higher than what one would expect from random sampling of the observed distribution — may indicate under-diversity: the generator clustering around the centre of the distribution rather than exploring the tails.

CTGAN falls well under the first threshold, and consistently so: fewer than half of its generated fish pass the K_area, width/length and thickness/length checks on average, and the spread across the 20 runs (26.5–51.2%) never approaches the 80% threshold even at its best — this is a stable property of the model, not an unlucky configuration. It is also the direct explanation for the sample-size gap noted in §5.1: at the 100% proportion, only 355 of 835 generated CTGAN rows survived the filter, against 764–822 for the other two synthesisers, meaning CTGAN's fidelity checks are run on the smallest and least statistically powered sample of the three — for the model whose output most needs scrutiny. This low pass rate is now the clearest standing evidence of CTGAN's biological implausibility; unlike an earlier pass of this analysis, it can no longer be attributed jointly to a `length_cm` marginal distortion, since §5.1 does not reproduce that finding on this re-run — the explanation narrows instead to the collinearity collapse documented in §5.2 (Table 5.2c): with only 167 training rows, CTGAN's adversarial training has learned to sample each morphometric dimension roughly correctly in isolation, but not the joint, physically-constrained combinations of length, width and thickness that the biological validity filter actually checks.

TVAE sits at the opposite extreme, and its near-ceiling pass rate (98.4%, reaching 100% in at least one run) is not an unambiguous positive. Given the "consistently above 95%" caution above, it is worth treating as an open question rather than a second confirmation of TVAE's quality: a model that rarely produces an invalid fish could be learning the true biological constraints well, or it could be generating conservative, overly central samples that rarely reach the geometric boundaries in the first place — which would also explain, independently of good calibration, why TVAE's marginals matched real training data so closely in §5.1. Distinguishing these two explanations needs the diversity checks of §5.6 (coverage of sparsely populated regions, internal distance distributions) rather than the pass rate alone.

GaussianCopula sits in the more comfortable middle (91.0%, tightly reproduced at ±2.9%), consistent with §5.1's read of it as distorting the *shape* of three marginals without pushing many samples outside biologically plausible bounds.

### 5.4 Preliminary TSTR Benchmark

The evaluation so far has been descriptive. The question that ultimately matters is simpler and harder: **can a model trained exclusively on synthetic data predict the weight of real, unseen fish as accurately as a model trained on real data?**

We answer this question with a preliminary TSTR (Train on Synthetic, Test on Real) experiment using 5-fold cross-validation on the real training set as the evaluation loop. The sealed real test set is not touched.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["length_cm", "width_cm", "thickness_cm"]
TARGET   = "weight_g"
KF       = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_allometric(X_train, y_train, X_val, y_val):
    """
    Fit log-log OLS allometric model; evaluate in original scale with Duan smearing.
    """
    X_tr = np.log(X_train)
    y_tr = np.log(y_train)
    model = LinearRegression().fit(X_tr, y_tr)

    smearing = np.mean(np.exp(y_tr - model.predict(X_tr)))   # Duan 1983

    y_pred = np.exp(model.predict(np.log(X_val))) * smearing
    return {
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2":  r2_score(y_val, y_pred),
    }

# ── TRTR baseline ──────────────────────────────────────────────────────────
X_real = df_train[FEATURES].values
y_real = df_train[TARGET].values

trtr_scores = [
    evaluate_allometric(X_real[tr], y_real[tr], X_real[va], y_real[va])
    for tr, va in KF.split(X_real)
]
df_trtr = pd.DataFrame(trtr_scores)
print(f"TRTR | MAE {df_trtr['MAE'].mean():.3f} ± {df_trtr['MAE'].std():.3f} "
      f"| R² {df_trtr['R2'].mean():.3f}")

# ── TSTR per synthesiser and proportion ───────────────────────────────────
tstr_records = []

for entry in synthetic_registry:
    df_valid = entry["df"][entry["df"]["bio_valid"]]
    if len(df_valid) < 20:
        continue

    X_synth = df_valid[FEATURES].values
    y_synth = df_valid[TARGET].values

    fold_scores = [
        evaluate_allometric(X_synth, y_synth, X_real[va], y_real[va])
        for _, va in KF.split(X_real)
    ]
    df_fold = pd.DataFrame(fold_scores)
    tstr_records.append({
        "model":      entry["model"],
        "proportion": entry["proportion"],
        "replica":    entry["replica"],
        "MAE_mean":   df_fold["MAE"].mean(),
        "MAE_std":    df_fold["MAE"].std(),
        "R2_mean":    df_fold["R2"].mean(),
    })

df_tstr = pd.DataFrame(tstr_records)
print(df_tstr.groupby("model")[["MAE_mean", "R2_mean"]].mean().round(3))
```

*Table 5.4a — TRTR baseline (5-fold CV on real training data, mean ± std across folds)*

| Metric | Value |
|---|---|
| MAE | 0.352 ± 0.083 g |
| R² | 0.973 ± 0.015 |

This reproduces Part 1's baseline (MAE = 0.352 g, R² = 0.973) essentially exactly — a useful sanity check that this notebook's CV pipeline is consistent with Part 1's, before trusting any comparison built on top of it.

*Table 5.4b — TSTR, mean across 5 replicas per model and proportion*

| Model | Proportion | MAE (g) | R² |
|---|---|---|---|
| TVAE | 25% | 0.667 | 0.893 |
| TVAE | 50% | 0.493 | 0.948 |
| TVAE | 100% | 0.519 | 0.938 |
| TVAE | 200% | 0.561 | 0.927 |
| GaussianCopula | 25% | 0.617 | 0.914 |
| GaussianCopula | 50% | 0.597 | 0.916 |
| GaussianCopula | 100% | 0.538 | 0.932 |
| GaussianCopula | 200% | 0.506 | 0.940 |
| CTGAN | 25% | 2.177 | 0.169 |
| CTGAN | 50% | 1.679 | 0.506 |
| CTGAN | 100% | 1.185 | 0.755 |
| CTGAN | 200% | 1.542 | 0.553 |

*Table 5.4c — Overall TSTR (mean across all proportions and replicas) vs. TRTR*

| Source | Mean MAE (g) | Ratio to TRTR |
|---|---|---|
| TRTR baseline | 0.352 | 1.00× |
| TVAE | 0.560 | 1.59× |
| GaussianCopula | 0.564 | 1.60× |
| CTGAN | 1.513 | 4.30× |

**Reading the results.** The headline finding confirms what §5.2 predicted before this benchmark was even run: none of the three synthesisers reaches TRTR. Training on synthetic data alone, however good, does not yet match training on the 167 real fish — the question §5.4 was designed to ask has a clear preliminary answer, and it is no.

The severity, though, is not uniform, and it lines up with §5.2's mechanistic diagnosis less cleanly than the previous pass suggested. CTGAN is still by far the worst performer in mean terms — roughly 4.3× worse in MAE than TRTR (1.51 g vs. 0.35 g) — but its R² no longer sits in a narrow, uniformly poor band: it ranges from 0.169 at 25% proportion to 0.755 at 100%, the single best R² of any synthesiser at any proportion in this table, GaussianCopula and TVAE included. That volatility is itself consistent with §5.2's diagnosis (a generator whose synthetic fish have lost the co-scaling relationship the allometric model depends on, §5.2's VIF ≈ 1 finding): with collinearity removed, CTGAN's coefficient estimates are essentially unconstrained, so how well any one training/sampling draw happens to transfer to real covariates can swing widely — a pattern §5.2's replica-level exponent instability (Table 5.2b) already anticipated. GaussianCopula and TVAE are both clearly short of TRTR, and now by almost the same margin (1.60× and 1.59× MAE respectively, against 1.60× and 1.34× in the previous pass) — TVAE's edge over GaussianCopula has largely disappeared on this re-run. Both stay in a practically different regime from CTGAN, though not as uniformly as before: R² stays above 0.91 for GaussianCopula at every proportion, but TVAE dips to 0.893 at 25% — still degraded rather than destroyed, but the two synthesisers are no longer as cleanly separable from each other as the original pass implied.

Unlike the previous pass, TVAE no longer beats GaussianCopula at every proportion — the ranking flips with the augmentation proportion. TVAE has lower MAE and higher R² at 50% (0.493 g / 0.948 vs. 0.597 g / 0.916) and 100% (0.519 g / 0.938 vs. 0.538 g / 0.932), but GaussianCopula is ahead at 25% (0.617 g / 0.914 vs. 0.667 g / 0.893) and, more markedly, at 200% (0.506 g / 0.940 vs. 0.561 g / 0.927). Averaged across proportions the two are now within 0.004 g of each other (Table 5.4c) — a gap this small should not be read as a real ranking between the two synthesisers without the paired statistical comparison, across folds and replicas, that §11 is designed to provide. On the evidence in this section alone, TVAE and GaussianCopula are best described as indistinguishable in TSTR performance, not TVAE-ahead-of-GaussianCopula as the earlier pass concluded. That reversal does not by itself overturn §5.2's diagnostic picture (GaussianCopula more internally coherent in-sample; TVAE's marginals more faithful to real covariate ranges) — it is a reminder that in-sample fidelity checks and out-of-sample TSTR performance can disagree, and that a preliminary CV benchmark on n=167 is not powered to detect a ranking this close in the first place. Any explanation resting on *why* one might edge out the other — e.g. how the Duan smearing correction interacts with each model's residual distribution — remains a hypothesis prompted by the data, not a conclusion it establishes, and needs a dedicated follow-up rather than a re-reading of the numbers already in hand.

The effect of augmentation proportion is itself informative, and only partly unchanged by the re-run. GaussianCopula still improves nearly monotonically as more synthetic rows are added (MAE 0.617 → 0.597 → 0.538 → 0.506 g from 25% to 200%; R² 0.914 → 0.916 → 0.932 → 0.940) — the ordinary variance-reduction effect of more training data, holding the generator's underlying bias roughly constant; these figures are unchanged from the previous pass, consistent with GaussianCopula's deterministic fit. TVAE still shows a similar, noisier improvement peaking at 50% rather than 200% (0.667 → 0.493 → 0.519 → 0.561 g), though again no per-proportion std is reported here to judge whether that ordering is meaningful or within noise. CTGAN shows no clean trend at all, and less so than before — its MAE now bounces between 1.19 g (its best result, at the 100% proportion, where R² also peaks at 0.755) and 2.18 g (its worst, at 25%), with no monotonic relationship to proportion. That volatility is hard to reconcile with a simple "more rows reduce sampling variance" story, and reinforces §5.2's point that CTGAN's exponent estimates, with collinearity collapsed, are not just biased but essentially unconstrained by the training data — more synthetic rows change how much unconstrained data the downstream model sees, not whether the underlying relationship it encodes is correct.

One more result is worth flagging without over-interpreting it, and it is more pronounced on this re-run than before: CTGAN's TSTR R² varies enormously with proportion (0.169 to 0.755) around its own in-sample allometric R² from §5.2 at the matching 100% proportion (0.361, Table 5.2c). At the 100% proportion specifically, TSTR R² (0.755) is more than double the in-sample figure; at 25% proportion, TSTR R² (0.169) is well below it. Given the two are measuring different things — in-sample fit on synthetic data's own covariate distribution vs. out-of-sample prediction on real covariates, at different proportions besides — this is not necessarily inconsistent, but the size of the swing, and the fact its direction reverses across proportions, means it should not be taken at face value without checking, in Part 3, whether either number holds up outside this preliminary CV loop and against the sealed test set.

TSTR is a *necessary but not sufficient* condition for useful augmentation, and this preliminary result is a negative one on its own terms: no synthesiser, used alone, replaces real data for this task. Part 3 will test the more practically relevant question this section could not — does *combining* real and synthetic data improve over real data alone, for any synthesiser, at any proportion — on the sealed real test set that has not been touched since Part 1.

---

## 6. What the Numbers Will Tell Us

By the end of running `part2_synthetic_generation.ipynb` you will have:

- Three fitted synthesisers, each producing biologically validated synthetic fish.
- A library of synthetic datasets covering four augmentation proportions and five independent replicas per configuration — 60 files in total (3 synthesisers × 4 proportions × 5 replicas).
- A univariate and multivariate fidelity report comparing each synthesiser's output to the real training distribution.
- A comparison of allometric exponents (b₁, b₂, b₃) between real and synthetic data.
- A preliminary TSTR benchmark establishing whether synthetic data, *on its own*, approximates the predictive performance of real data.

The results of these checks determine the agenda for Part 3. If all three synthesisers pass biological validity at high rates and reproduce the allometric structure faithfully, Part 3 can proceed directly to the augmentation experiments. If one synthesiser systematically fails — by distorting correlations or generating biologically implausible fish at scale — it will be flagged and either excluded or treated as a negative reference case.

---

## 7. Coming in Part 3

Part 3 closes the loop. The sealed real test set — untouched since `part1_eda_v2.ipynb` created it — is opened exactly once to evaluate the full range of strategies:

- **TRTR**: the real-only baseline.
- **TSTR**: synthetic-only, per synthesiser and proportion.
- **Hybrid augmentation**: real training data extended with synthetic records at 25%, 50%, 100% and 200%.
- **Statistical comparison**: are the observed MAE differences consistent across partitions and seeds, or within the noise floor of the experiment?

The threshold from Part 1 is clear: any strategy claiming to improve over the baseline must achieve MAE < 0.352 g and R² > 0.973 on the sealed real test set, with a statistically consistent advantage across at least five independent replicas.

We will also ask the harder question: even if synthetic data does not improve the average, does it help in the regions where the real training set is sparsest — the upper morphometric tail? That answer requires the sealed test set, and it requires the careful experimental infrastructure built across Parts 1 and 2.

---

*→ [Part 3: Does Augmentation Actually Help? Statistical Evaluation on Real Test Data](#)*

---

## References

- Bolger, T., & Connolly, P. L. (1989). The selection of suitable indices for the measurement and analysis of fish condition. *Journal of Fish Biology*, 34(2), 171–182. https://doi.org/10.1111/j.1095-8649.1989.tb03300.x
- Costas, B., Aragão, C., Mancera, J. M., Dinis, M. T., & Conceição, L. E. C. (2008). High stocking density induces crowding stress and alters the physiological and immune status of juvenile *Solea senegalensis*. *Aquaculture*, 230(1–4), 1–10. https://doi.org/10.1016/j.aquaculture.2012.11.019
- Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017
- Froese, R. (2006). Cube law, condition factor and weight–length relationships: History, meta-analysis and recommendations. *Journal of Applied Ichthyology*, 22(4), 241–253. https://doi.org/10.1111/j.1439-0426.2006.00805.x
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*. https://arxiv.org/abs/1312.6114
- Le Cren, E. D. (1951). The length-weight relationship and seasonal cycle in gonad weight and condition in the perch (*Perca fluviatilis*). *Journal of Animal Ecology*, 20(2), 201–219. https://doi.org/10.2307/1540
- Nelson, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer. https://link.springer.com/book/10.1007/0-387-28678-0
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1907.00503
