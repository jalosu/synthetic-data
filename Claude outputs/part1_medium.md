# Beyond the Scale: Can Synthetic Data Solve the Small-Dataset Problem in Aquaculture AI?
## Part 1: The Data Challenge, Biometrics & Allometric Baselines

*Part 1 of a 3-part series on synthetic data for allometric weight estimation in precision aquaculture.*

---

## 1. Hook: The Grading Problem in Precision Aquaculture

Imagine you are standing in front of a tank holding 50,000 juvenile sole — each one roughly the size of a large coin. In the next few hours, every single one of them needs to be sorted by weight and redistributed into homogeneous groups. This process is called **grading**, and in commercial flatfish aquaculture it is not optional: fish of different sizes compete aggressively for food, and the larger ones consistently win. Without grading, growth diverges, feed conversion deteriorates, and mortality climbs. Done right, grading is one of the highest-leverage interventions a fish farm can make.

The problem is that weighing 50,000 fish individually — by hand, one at a time — is not only impractical. It causes measurable physiological damage. Every second a juvenile sole spends out of water triggers a cortisol stress response that suppresses immune function, disrupts osmotic balance, and can increase post-handling mortality for days afterward. The animals you are trying to sort are also the animals you are inadvertently harming in the process.

This is exactly the problem that **GRADIAX** is trying to solve: **replace manual weighing with real-time computer vision**. The system captures images of individual fish passing through a classification conveyor, extracts morphometric measurements — length, width, dorsal thickness — and estimates weight instantly, without any contact. Thousands of fish per hour, zero handling stress.

The architecture is elegant. The challenge, as always in applied machine learning, is the data.

To train a model that reliably maps morphometric measurements to body weight, you need a dataset that is representative, clean, and large enough to cover the full morphological space of the species — including the extremes. And acquiring that dataset requires exactly what you are trying to avoid: careful, ethical, one-by-one measurement of live fish under controlled conditions.

This is the paradox at the heart of precision aquaculture AI, and it is the subject of this three-part series. In Part 1, we establish the ground truth: what the real data looks like, what it tells us biologically, and how well classical allometric models perform on it. In Part 2, we generate synthetic fish using three different generative models and rigorously evaluate whether the fakes are biologically credible. In Part 3, we answer the question that actually matters: **does training with synthetic data improve weight estimation on real, unseen fish — and if so, by how much, under what conditions, and with which model?**

Let's start with the fish.

---

## 2. Why You Can't Just Collect More Data

The obvious solution to a small dataset is to collect more data. In most machine learning domains this advice is correct and actionable. In juvenile flatfish research, it runs headlong into a hard biological constraint.

*Solea senegalensis* — Senegalese sole — is one of the most commercially valuable species in Southern European aquaculture, and one of the most physiologically sensitive to handling stress. Studies by [Costas et al. (2013)](https://doi.org/10.1016/j.aquaculture.2012.11.019) have shown that repeated manipulation of juvenile sole induces acute cortisol elevation, impairs osmoregulation, and increases post-handling mortality in a dose-dependent manner. Each additional measurement on the same individual is not a free observation: it is a welfare cost that accumulates.

The implication for data collection is direct. Measurement sessions must be:

- **Short** — each fish out of water for no more than 20 seconds, on a wet surface to prevent mucus loss
- **Single-pass** — no repeat measurements of the same individual
- **Conducted under controlled sedation** — light anaesthesia to reduce movement artefacts without causing physiological harm
- **Supervised by a trained single operator** — to minimise inter-observer variability

These constraints are not bureaucratic inconveniences. They are the reason the dataset in this study contains 209 records rather than 2,000. And they are representative of a broader class of biological datasets where data quality and data quantity are in direct tension.

The experimental protocol was conducted in compliance with EU Directive 2010/63/EU on the protection of animals used for scientific purposes. Fish were acclimated for 24 hours in quarantine tanks at 18 ± 1°C, salinity 35 ± 1 ppt, dissolved oxygen > 6 mg/L, and fasted for 12 hours prior to measurement to eliminate weight variation from gut content. Each individual was measured for total length (rostrum to caudal fin tip), maximum body width, maximum dorsoventral height, and wet body weight, using a calibrated digital balance (±0.01 g) and a digital calliper (±0.01 cm).

The result is a small but methodologically rigorous dataset. High individual quality, limited statistical breadth. This is precisely the scenario where synthetic data augmentation is most often proposed — and least often rigorously evaluated.

> **Key point for practitioners:** Before reaching for a generative model to solve a small-data problem, always ask *why* the dataset is small. The answer determines what kinds of synthetic augmentation are biologically and statistically defensible. In this case, the constraint is ethical and practical, not economic — which means the synthetic data must respect the same biological constraints that made collection difficult in the first place.

---

## 3. The Dataset: 209 Sole, 4 Variables, Zero Compromises

The dataset produced by this protocol is deliberately minimal in structure. Four morphometric variables, 209 individuals, no repeated measurements, no missing values.

| Variable | Type | Unit | Description |
|---|---|---|---|
| Weight | Numeric | g | Wet body weight |
| Length | Numeric | cm | Total length (rostrum → caudal fin) |
| Width | Numeric | cm | Maximum transverse width |
| Thickness | Numeric | cm | Maximum dorsoventral thickness |

The summary statistics below already reveal several biologically meaningful patterns before any modelling takes place.

| Statistic | Weight (g) | Length (cm) | Width (cm) | Thickness (cm) |
|---|---|---|---|---|
| Mean | 5.34 | 7.24 | 2.79 | 0.46 |
| Std | 3.64 | 1.51 | 0.70 | 0.12 |
| Min | 0.46 | 3.30 | 1.10 | 0.20 |
| 25% | 2.82 | 6.10 | 2.30 | 0.40 |
| Median | 4.29 | 7.00 | 2.70 | 0.50 |
| 75% | 6.96 | 8.30 | 3.30 | 0.50 |
| Max | 21.98 | 11.40 | 5.20 | 0.90 |

Three things stand out immediately. First, **weight has a coefficient of variation (~68%) more than three times larger than length (~21%) — a ratio of 3.3×**, which is consistent with allometric growth: small absolute differences in linear dimensions translate into large volumetric — and therefore mass — differences. Second, **thickness is remarkably compressed**: the interquartile range spans only 0.1 cm (0.4–0.5 cm), and the 75th and 50th percentiles are identical. This is a structural feature of flatfish morphology — dorsoventrally compressed bodies don't vary much in thickness — and it has direct implications for which variables will carry predictive signal. Third, the **maximum weight (21.98 g) is more than twice the 75th percentile (6.96 g)**, suggesting a right skew and the presence of a small number of notably larger individuals, whether biologically exceptional or potential measurement outliers.

These are not just descriptive footnotes. They define the constraints any generative model must respect: synthetic fish that are heavier than they are long, wider than they are long, or with negative dimensions are biologically impossible. We will use these constraints explicitly in Part 2 to filter synthetic samples.

![Morphometric Space of S. senegalensis Juveniles](fig1_morphometric_space.png)
*Each point is one fish. Bubble size encodes body width; colour encodes dorsoventral thickness. The curvilinear dispersion — not a straight line — is the signature of allometric growth. Figure generated from simulated data replicating the dataset's summary statistics; run `part1_eda_v2.ipynb` to obtain the equivalent plot from real training observations.*

> **A note on scope:** All 209 individuals come from a single cohort of 90-day post-hatch juveniles in a single production tank. This is important. The models we train in Part 3 are measuring *interpolation within a narrow developmental window*, not generalisation across ages, seasons, farms, or genetic lines. Synthetic augmentation cannot invent biological variability that was never observed. We will return to this limitation in the conclusions.

---

## 4. Exploratory Data Analysis — What the Fish Tell Us

Before fitting any model, we need to understand the structure of the data: how each variable is distributed, how variables relate to one another, where the nonlinearities live, and whether there are observations that deserve scrutiny. This section walks through the full EDA. All code is available in the companion notebook **`part1_eda_v2.ipynb`**.

### 4.1 Univariate Distributions

Weight follows a right-skewed unimodal distribution — most fish cluster between 2 and 7 g, with a long tail extending to ~22 g. This asymmetry is biologically expected: growth in fish populations is not symmetric, and the largest individuals in any cohort tend to be disproportionately heavy relative to the rest. The practical implication for modelling is that ordinary least squares regression will underperform in the upper weight range unless the target is transformed or a robust loss function is used.

Length is approximately normal and symmetric, which reflects a well-managed cohort with consistent age and feeding conditions. Width mirrors length closely. Thickness, as noted above, shows a near-discrete distribution concentrated around 0.4–0.5 cm — a direct consequence of the flatfish body plan.

### 4.2 Bivariate Relationships and Allometric Structure

The pairplot reveals the core story of this dataset: **weight grows non-linearly with length and width**, tracing curvilinear clouds that become increasingly dispersed as fish get larger. This is the hallmark of allometric growth — the same biological phenomenon that makes a whale not simply a scaled-up salmon. A linear model fitted to these relationships will systematically underestimate heavy fish and overestimate light ones.

Length and width are tightly correlated (r ≈ 0.94), which is expected in a single cohort with stable proportions — a property known as *planiform isometry* in flatfish morphometrics. This colinearity has modelling consequences: using both variables in a linear regression without regularisation will inflate coefficient variance. Ridge regression and allometric log-linearisation both handle this more gracefully than ordinary least squares.

### 4.3 Correlation Structure: Pearson vs. Spearman

We compute both Pearson and Spearman correlations. The difference matters: Pearson measures linear association, while Spearman captures monotonic relationships including the nonlinear allometric ones that dominate this dataset.

The Spearman–Pearson difference matrix reveals the structure clearly. **Weight–Length and Weight–Thickness show the largest nonlinear components** (Δ ≈ 0.055 and 0.052 respectively): both relationships are strongly monotonic but contain substantial curvature — exactly what allometric theory predicts. **Weight–Width shows a much smaller gap (Δ = 0.015)**, indicating that once length is accounted for, width tracks weight more proportionally. The **Length–Width pair has a near-zero gap (Δ = 0.009)**, confirming that these two dimensions scale almost linearly with each other within this cohort — the source of the multicollinearity discussed in Section 4.2.

### 4.4 Heteroscedasticity and the Case for the Multivariate Allometric Model

Plotting residuals from a naive linear regression of weight on length makes the problem concrete: the variance of the residuals grows with fitted values. Heavier fish are harder to predict not because the relationship is weaker, but because the absolute scale of variation increases with size. This heteroscedasticity violates the assumptions of ordinary least squares and inflates error in the upper weight range — precisely where grading decisions matter most (large fish are worth more and their misclassification has higher economic consequences).

The standard response in fisheries biology is to fit the univariate power law W = a·L^b, log-linearised as log(W) = log(a) + b·log(L). This model has been the field reference for decades (Froese 2006; Le Cren 1951) and its exponent b is directly interpretable: b = 3 indicates isometric growth, b < 3 negative allometry (the fish gets flatter as it grows), b > 3 positive allometry. For *S. senegalensis*, we expect b < 3 precisely because the dorsoventral axis is constrained.

However, the univariate model discards two dimensions we have already measured. Because body weight is proportional to volume, and volume scales as L · A · E, a more complete — and physically better justified — model is the multivariate allometric power law:

**W = a · L^b₁ · A^b₂ · E^b₃**

Each exponent now captures the differential allometric contribution of its dimension. In *S. senegalensis*, thickness grows far more slowly than length and width during the juvenile phase, so b₃ will be substantially smaller than b₁ and b₂ — a distinction the univariate model collapses into a single exponent and consequently cannot represent.

There is a subtler problem with the univariate model that goes beyond prediction accuracy. Because length and width are tightly correlated in a single cohort (r ≈ 0.94), the univariate exponent b absorbs part of the width signal as well as the length signal — a classic case of **omitted variable bias**. Formally, b_observed ≈ b₁ + b₂ · (∂ log A / ∂ log L). On the real dataset, this yields b = 3.016 — marginally above the isometric expectation of 3 and inconsistent with the b₁+b₂+b₃ = 2.89 recovered by the multivariate model (which correctly attributes negative allometry, i.e. sub-cubic growth, to this dorsoventrally compressed species). The univariate exponent is not a catastrophically wrong number, but it cannot be interpreted as a pure length-weight allometric coefficient: it is a composite that conflates the contributions of length and width and therefore lacks independent biological meaning.

Studies in related flatfish species consistently show meaningful predictive improvements when moving from the univariate to the multivariate form (Froese 2006). We include both models in Section 5 to quantify this gain empirically on our own data.

Log-transformation linearises this model naturally:

**log(W) = log(a) + b₁·log(L) + b₂·log(A) + b₃·log(E)**

This is standard multiple linear regression in log space, which simultaneously stabilises residual variance, linearises the allometric relationship, and yields directly interpretable exponents for each dimension. One critical detail that allometric modellers routinely overlook: retransforming predictions from log scale back to grams introduces a systematic negative bias. The correction requires multiplying predictions by Duan's smearing factor — the mean of the exponentiated log-space residuals — which we compute and apply explicitly in Section 5.

![From heteroscedasticity to allometric precision — three models compared](fig2_ols_vs_allometric.png)
*Each panel shows observed vs. predicted weight (g); the dashed line is the 1:1 reference. Colour encodes relative residual magnitude (green = low, red = high). Points are simulated from the real fitted parameters to illustrate model behaviour; all annotated coefficients (b, b₁, b₂, b₃, R²) are from fitting on real training data (n = 167 observations). **(A) Linear OLS** — residuals fan out with fish size, large fish are systematically underestimated. **(B) Univariate W = a·L^b** — log-transformation stabilises variance and yields b = 3.016, marginally above 3. This mild positive bias is consistent with omitted variable bias: with r ≈ 0.94 between length and width, the univariate exponent absorbs part of the width signal (b_obs ≈ b₁ + b₂·∂log A/∂log L). The multivariate model is needed to disentangle the individual contributions. **(C) Multivariate W = a·L^b₁·A^b₂·E^b₃** — partitioning the signal recovers individually interpretable exponents (b₁ = 1.62, b₂ = 0.88, b₃ = 0.39; b₁+b₂+b₃ = 2.89 < 3, consistent with negative allometry in a dorsoventrally compressed species) and raises R² from 0.959 to 0.982 in log space.*

### 4.5 Biological Validity Checks

Finally, we verify that the real data itself is internally consistent before using it to train any generative model. This step is often skipped and should not be. Key checks:

- All dimensions strictly positive ✓
- Weight within plausible range for 90-day *S. senegalensis* juveniles (0.3–25 g) ✓
- Length > Width > Thickness in all records ✓ (flatfish anatomical ordering)
- Width/Length ratio within [0.20, 0.55] — **1 record flagged** (index 153: L=6.8 cm, A=3.9 cm, ratio=0.574)
- Thickness/Length ratio within [0.02, 0.12] ✓ (flatness index, appropriate for dorsoventrally compressed fish)
- Planiform condition index K_area = W/(L×Width) within [0.08, 0.50] g/cm² — replaces Fulton's K, which assumes isometric 3D growth (b = 3) and is **invalid for flatfish** (Froese 2006; Le Cren 1951; Bolger & Connolly 1989)

In total, **1 out of 209 records (0.5%) is flagged as bio_valid = False**, due to an unusually wide body relative to length (width/length = 0.574, above the 0.55 upper bound for typical planiform proportions). The individual is otherwise internally consistent — weight, thickness and K_area fall within expected ranges. Rather than removing it, we retain it with the `bio_valid` flag set to False and track its influence through the pipeline. An atypically wide fish is not necessarily a measurement error; it may reflect genuine morphological variation at the extremes of the distribution.

---

## 5. Allometric Baselines: How Well Can We Do With Real Data Alone?

Before introducing any synthetic data, we need a rigorous answer to a simpler question: what is the best weight estimate achievable with the 167 real training observations we have? This baseline is not just a starting point — it is the scientific standard against which everything in Part 3 will be judged. A synthetic augmentation strategy that does not outperform this baseline is not useful, regardless of how statistically similar the fake fish look to the real ones.

We evaluate two allometric models of increasing complexity, plus a linear OLS baseline for reference. The design is deliberate: the univariate power law W = a·L^b is the canonical reference in fisheries science and must appear for comparability with the literature. The multivariate form W = a·L^b₁·A^b₂·E^b₃ is what the physics suggests is correct. Fitting both allows us to quantify empirically how much width and thickness actually contribute on *this* dataset — an answer that cannot be assumed from theory alone. In Part 3, we will also need both baselines to assess whether synthetic augmentation helps more in the single-dimension or multi-dimension regime.

All models are fitted exclusively on the training set and evaluated using **5-fold cross-validation** — the test set remains sealed. This is critical: using the test set for baseline evaluation and then again for the augmented models would make the comparison meaningless.

### 5.1 Model 1 — Univariate Power Law: W = a · L^b

The simplest allometric model relates weight to a single linear dimension via a power law, fitted via log-linearisation on the training set:

```
log(W) = log(a) + b · log(L)
→  a = 0.01194,  b = 3.0161
   R² (log space) = 0.959,  Duan smearing factor = 1.009
```

The estimated exponent b = 3.016 is marginally above the isometric expectation of 3. As discussed in Section 4.4, this is consistent with omitted variable bias: with r ≈ 0.94 between length and width, the univariate exponent absorbs part of the width signal (b_obs ≈ b₁ + b₂ · ∂log A/∂log L). The multivariate model in Section 5.2 will decompose this into its independent components and confirm that the true allometric sum b₁+b₂+b₃ < 3, consistent with the negative allometry expected for a dorsoventrally compressed species.

The Duan smearing factor of 1.009 indicates a small but non-negligible retransformation bias: predictions from the log-space model must be multiplied by this factor before comparison in grams.

### 5.2 Model 2 — Multivariate Power Law: W = a · L^b₁ · A^b₂ · E^b₃

The multivariate model uses all three measured dimensions, fitted in log-log space:

```
log(W) = log(a) + b₁·log(L) + b₂·log(A) + b₃·log(E)
→  a = 0.1050
   b₁ (Length)    = 1.621
   b₂ (Width)     = 0.876
   b₃ (Thickness) = 0.393
   R² (log space) = 0.982,  Duan smearing factor = 1.004
```

Three results stand out from these exponents. First, **b₁+b₂+b₃ = 2.890 < 3**, confirming negative allometry at the whole-body level — consistent with the flatfish body plan and resolving the apparent b ≈ 3.0 of the univariate model into its true components. Second, **b₁ = 1.621 is lower than conventional wisdom for length alone** (often assumed ~3 in univariate contexts), but this is expected: once width and thickness are in the model, length is no longer acting as a proxy for volume — it contributes only its true geometric share. Third, **b₃ = 0.393 for thickness is non-trivial despite its narrow absolute range (IQR = 0.1 cm)**. The log-space regression isolates the contribution of each dimension independently: even modest thickness variation captures information about body condition and fat content that neither length nor width encodes directly, likely reflecting individual differences in nutritional state within the cohort.

The Duan smearing factor of 1.004 is very close to 1, indicating that the log-space model is nearly unbiased on retransformation — the residuals in log space are symmetric and the correction is practically negligible, though still applied for rigour.

> **Reference:** Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610. [https://doi.org/10.2307/2288126](https://doi.org/10.2307/2288126)

### 5.3 Cross-Validated Performance

The table below summarises 5-fold cross-validation results on the training set. All metrics are computed in grams on the original scale after applying the Duan smearing correction.

| Model | MAE (g) ± std | RMSE (g) ± std | R² ± std | Median AE (g) |
|---|---|---|---|---|
| Linear OLS (W ~ L + A + E) | 0.666 ± 0.093 | 1.043 ± 0.299 | 0.902 ± 0.017 | 0.434 |
| Univariate W = a·L^b | 0.535 ± 0.111 | 0.893 ± 0.314 | 0.927 ± 0.025 | 0.317 |
| **Multivariate W = a·L^b₁·A^b₂·E^b₃** | **0.352 ± 0.075** | **0.523 ± 0.135** | **0.973 ± 0.014** | **0.243** |

*5-fold cross-validation on training set (n ≈ 167). Test set remains sealed. Duan smearing correction applied to all log-space models.*

The multivariate allometric model consistently outperforms both baselines across all folds. The gain over the univariate is substantial — MAE drops from 0.535 g to 0.352 g (−34%) and R² rises from 0.927 to 0.973 — confirming that width and thickness carry independent predictive signal beyond what length alone captures. The standard deviations across folds are moderate, indicating stable behaviour under different data partitions.

> **This is the number that matters for Part 3.** Any synthetic augmentation strategy that does not reduce MAE below **0.352 g** or improve R² beyond **0.973** on the held-out test set provides no practical benefit over simply using the 167 real training observations with a multivariate allometric model — regardless of how well the generative model scores on distributional similarity metrics.

### 5.4 The Open Question

The multivariate allometric model is already a strong baseline for 167 training observations from a single cohort. But two structural limitations remain. First, the model has seen only one cohort — its generalisation to new batches, different tank conditions or broader size ranges is unknown. Second, the upper tail of the weight distribution is sparsely sampled: with only ~10% of individuals above the 90th percentile (~12 g) and the dataset maximum at 21.98 g, predictions for the heaviest fish rest on very few observations. This is precisely the region where grading decisions are most consequential economically.

These are exactly the two scenarios where synthetic augmentation is most often proposed as a remedy. In Part 2 we generate the synthetic fish. In Part 3 we find out whether they actually help.

---

*Continue to **Part 2: Generating Fake Fish** — GaussianCopula, CTGAN and TVAE in action.*

---

### References

- Bolger, T. & Connolly, P. L. (1989). The selection of suitable indices for the measurement and analysis of fish condition. *Journal of Fish Biology*, 34(2), 171–182. https://doi.org/10.1111/j.1095-8649.1989.tb03300.x
- Costas, B., et al. (2013). Physiological responses of Senegalese sole (*Solea senegalensis*) to acute stress. *Aquaculture*, 416–417, 56–63. https://doi.org/10.1016/j.aquaculture.2012.11.019
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017
- Froese, R. (2006). Cube law, condition factor and weight–length relationships: history, meta-analysis and recommendations. *Journal of Applied Ichthyology*, 22(4), 241–253. https://doi.org/10.1111/j.1439-0426.2006.00805.x
- Le Cren, E. D. (1951). The length-weight relationship and seasonal cycle in gonad weight and condition in the perch (*Perca fluviatilis*). *Journal of Animal Ecology*, 20(2), 201–219. https://doi.org/10.2307/1540
