# Veritas AI — Mathematical Formulations for Forensic AI Detection
## Rigorous Theoretical Derivations, Equations, and Inference Fusion (2024–2026)

**Author:** Veritas AI Research & Engineering  
**Version:** 1.0.0 (Mathematical Specifications)  
**Target Hardware Constraints:** $\le$ 1.5GB RAM, 2-Core CPU, Zero PyTorch at Runtime (Pure ONNX INT8 + NumPy)

---

## 1. Executive Overview

Standard first-generation AI detectors relied either on naive perplexity thresholds or raw supervised softmax scores from language models. These methods suffer from two catastrophic failure modes:
1. **Perplexity-Topic Conflation**: Technical human prose (e.g. medical, legal, scientific) naturally exhibits high perplexity, leading to high false-positive rates (FPR), particularly against non-native English (ESL) writers.
2. **Authorial Voice Degradation**: Highly creative human authors with rich, idiosyncratic vocabularies and varied punctuation (such as David Sedaris) are frequently misclassified by linear classifiers as "AI-refined" due to statistical vocabulary dispersion.

To overcome these failure modes, **Veritas AI** introduces a unified set of **seven closed-form mathematical equations** that operate across feature extraction, logit space fusion, and calibrated posterior probability calculation.

```
                          INPUT DOCUMENT (x)
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
   Transformer Neural Trunk                        Forensic Stylometry
(ONNX INT8 MiniLM Distilled)                     (20-Dimensional Vector)
          │                                               │
          ▼                                               ▼
     Logits z_k                               Mathematical Equations
(4-Class Neural Likelihood)               ├── Ω_lex (Lexical Richness)
          │                               ├── B_syntax (Syntactic Burstiness)
          │                               ├── Φ_disc (Discourse Polarity)
          │                               ├── R_binoc (Binoculars Ratio)
          │                               └── Λ_auth (Authorial Affinity)
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                     Bayesian Forensic Fusion
               s_k = W_meta · f_norm + b + z_k + g_k(x)
                                  │
                                  ▼
                     Temperature-Scaled Softmax
                       P*(y = k | x; T*)
                                  │
                                  ▼
                   Dynamic Uncertainty Gating
                             U(x)
```

---

## 2. The Core Mathematical Equations

### Equation 1: Length-Invariant Lexical Richness Equation ($\Omega_{\text{lex}}$)

Traditional Type-Token Ratio ($\text{TTR} = V / N$) inherently degrades as text length $N$ increases (the Herdan-Heaps law effect). We formulate the **Unified Authorial Lexical Richness Index** $\Omega_{\text{lex}}$, which fuses three length-invariant estimators:

$$\Omega_{\text{lex}}(x) = w_D \cdot D_{\text{Simpson}}(x) + w_R \cdot \tilde{R}_{\text{Honoré}}(x) + w_K \cdot \max\left(0, 1 - \frac{K_{\text{Yule}}(x)}{K_{\max}}\right)$$

Where the weights are set to $w_D = 0.40, w_R = 0.40, w_K = 0.20$, and $K_{\max} = 200.0$.

#### Constituent Formulations:
1. **Simpson's Diversity Index ($D_{\text{Simpson}}$)**:
   Measures the probability that two randomly sampled tokens belong to different lexical types:
   $$D_{\text{Simpson}} = 1 - \frac{\sum_{i=1}^V n_i (n_i - 1)}{N(N - 1)}$$
   Where $n_i$ is the frequency count of word type $i$, $V$ is vocabulary size, and $N = \sum n_i$ is total word tokens.

2. **Normalized Honoré's Statistic ($\tilde{R}_{\text{Honoré}}$)**:
   Quantifies the distribution of hapax legomena ($V_1$, words appearing exactly once) relative to vocabulary size $V$, normalized by $R_{\text{norm}} = 2500.0$:
   $$\tilde{R}_{\text{Honoré}} = \min\left(1.0, \frac{100 \ln N}{R_{\text{norm}} \cdot \left(1 - \min(0.99, V_1 / V)\right)}\right)$$

3. **Yule's Characteristic ($K_{\text{Yule}}$)**:
   Based on the negative binomial distribution of token occurrences:
   $$K_{\text{Yule}} = 10^4 \cdot \frac{\sum_{i=1}^V n_i^2 - N}{N^2}$$

---

### Equation 2: Multi-Scale Syntactic Burstiness & Curvature Equation ($\mathcal{B}_{\text{syntax}}$)

Autoregressive language models generate text with unnaturally uniform sentence length trajectories. Authentic human authors oscillate between punchy rhetorical clauses (3–8 words) and complex compound sentences (35–50 words).

We capture this through the **First- and Second-Order Syntactic Rhythm Equation**:

$$\mathcal{B}_{\text{syntax}}(x) = \alpha_1 \left( \frac{\Delta_{\text{cadence}}}{\mu(L)} \right) + \alpha_2 \left( \frac{\kappa_{\text{cadence}}}{\mu(L)} \right) + \alpha_3 \cdot \text{CV}_{\text{len}}$$

Where $\alpha_1 = 0.40, \alpha_2 = 0.30, \alpha_3 = 0.30$, and:

1. **First-Order Cadence Delta (Rhythm Velocity)**:
   $$\Delta_{\text{cadence}} = \frac{1}{S - 1} \sum_{s=1}^{S-1} |L_{s+1} - L_s|$$
   Where $L_s$ is the token count of sentence $s$, and $S$ is the total sentence count.

2. **Second-Order Cadence Curvature (Rhythm Acceleration)**:
   $$\kappa_{\text{cadence}} = \frac{1}{S - 2} \sum_{s=1}^{S-2} |L_{s+2} - 2L_{s+1} + L_s|$$

3. **Sentence Length Coefficient of Variation**:
   $$\text{CV}_{\text{len}} = \frac{\sigma(L)}{\mu(L)} = \frac{\sqrt{\frac{1}{S}\sum_{s=1}^S (L_s - \mu(L))^2}}{\frac{1}{S}\sum_{s=1}^S L_s}$$

- **Human Text**: $\mathcal{B}_{\text{syntax}} \ge 0.65$ *(UNVERIFIED HYPOTHESIS: uncalibrated threshold)* (dynamic cadence shifts and high second-order acceleration).
- **AI Text**: $\mathcal{B}_{\text{syntax}} \le 0.48$ *(UNVERIFIED HYPOTHESIS: uncalibrated threshold)* (flat, uniform pacing).

---

### Equation 3: Bounded Discourse Polarity Index Equation ($\Phi_{\text{disc}}$)

Frontier LLMs (GPT-4o, Claude 3.5, DeepSeek, Qwen) exhibit persistent transitional discourse markers ("furthermore", "moreover", "in conclusion", "it is worth noting", "crucial role", "a testament to"). Conversely, authentic humans frequently write in first-person narrative voice ("I", "my", "we", "us", "dad", "got", "actually", "kinda").

We map these frequencies into a strictly bounded polarity score $\Phi_{\text{disc}} \in [-1.0, +1.0]$:

$$\Phi_{\text{disc}}(x) = \frac{\rho_{\text{AI}}(x) - \rho_{\text{Hum}}(x)}{1.0 + \rho_{\text{AI}}(x) + \rho_{\text{Hum}}(x)}$$

Where:
- $\rho_{\text{AI}}(x) = \frac{\text{Count}(\text{AI Transition Markers and Multi-Word Phrases})}{N} \times 100\%$
- $\rho_{\text{Hum}}(x) = \frac{\text{Count}(\text{Authentic Human Personal and Conversational Markers})}{N} \times 100\%$

#### Mathematical Properties:
- As $\rho_{\text{Hum}} \gg \rho_{\text{AI}}$, $\Phi_{\text{disc}} \to -1.0$ (Deep human authorial voice).
- As $\rho_{\text{AI}} \gg \rho_{\text{Hum}}$, $\Phi_{\text{disc}} \to +1.0$ (Signature LLM discourse pattern).
- If neither or both are balanced, $\Phi_{\text{disc}} \approx 0.0$ (Neutral academic prose).

---

### Equation 4: Binoculars Information-Compression Ratio Equation ($\mathcal{R}_{\text{Binoc}}$)

Building on Hans et al. (ICML 2024), LLMs generate text that lies on low-entropy probabilistic paths. In our pure CPU runtime, we approximate this via the ratio of **Shannon Information Entropy** to **DEFLATE Lossless Physical Byte Compression**:

$$\mathcal{R}_{\text{Binoc}}(x) = \frac{\mathcal{H}_{\text{Shannon}}(x)}{C_{\text{Deflate}}(x)}$$

Where:
$$\mathcal{H}_{\text{Shannon}}(x) = -\sum_{w \in \mathcal{V}} p(w) \log_2 p(w), \quad p(w) = \frac{\text{count}(w)}{N}$$
$$C_{\text{Deflate}}(x) = \frac{|\text{Deflate}(x)|}{|x_{\text{bytes}}|}$$

Because LLM text exhibits formulaic subword sequences, DEFLATE (LZ77 + Huffman) achieves higher compression (lower $C_{\text{Deflate}} \approx 0.40 - 0.45$ *(UNVERIFIED HYPOTHESIS)*), causing $\mathcal{R}_{\text{Binoc}}$ to elevate ($\ge 16.5$ *(UNVERIFIED HYPOTHESIS)*). Human writing exhibits irregular collocations ($C_{\text{Deflate}} \ge 0.52$ *(UNVERIFIED HYPOTHESIS)*), keeping $\mathcal{R}_{\text{Binoc}} \le 14.0$ *(UNVERIFIED HYPOTHESIS)*.

---

### Equation 5: Forensic Authorial Affinity Index Equation ($\Lambda_{\text{auth}}$)

To combine syntactic variance, discourse polarity, and physical Kolmogorov compressibility into a single unified discriminant:

$$\Lambda_{\text{auth}}(x) = \beta_1 \cdot \mathcal{B}_{\text{syntax}}(x) - \beta_2 \cdot \Phi_{\text{disc}}(x) + \beta_3 \cdot C_{\text{Deflate}}(x)$$

With parameter weights $\beta_1 = 0.40, \beta_2 = 0.40, \beta_3 = 0.20$.

#### Decision Boundary *(UNVERIFIED HYPOTHESIS: theoretical cutoffs)*:
$$\begin{cases} 
\Lambda_{\text{auth}}(x) \ge 0.65 & \implies \text{High Authorial Humanity (Authentic Human Voice)} \\
0.35 \le \Lambda_{\text{auth}}(x) < 0.65 & \implies \text{Hybrid / AI-Polished Zone} \\
\Lambda_{\text{auth}}(x) < 0.35 & \implies \text{High LLM Generation Likelihood}
\end{cases}$$

---

### Equation 6: Bayesian Logit Fusion & Calibrated Posterior Equation ($P^*(y \mid x)$)

Let $\mathbf{z} = (z_0, z_1, z_2, z_3)^T \in \mathbb{R}^4$ be the pooled neural logits from the INT8 transformer across document chunks:
$$z_k = \frac{1}{C} \sum_{c=1}^C z_{c, k}$$

Let $\mathbf{f} \in \mathbb{R}^{20}$ be the normalized stylometric feature vector:
$$\tilde{\mathbf{f}} = \frac{\mathbf{f} - \boldsymbol{\mu}_f}{\boldsymbol{\sigma}_f}$$

The fused raw score vector $\mathbf{s} \in \mathbb{R}^4$ is:
$$\mathbf{s} = \mathbf{W}_{\text{meta}} \tilde{\mathbf{f}} + \mathbf{b}_{\text{meta}} + \mathbf{z}$$

We apply the closed-form forensic correction vector $\mathbf{g}(x) = (g_0, g_1, g_2, g_3)^T$:
$$g_0 = \begin{cases} 
0.8 + 0.3 \cdot \left(\frac{\rho_{\text{Hum}}}{2.0}\right), & \text{if } (s_0 > s_2 \land s_0 > s_3) \land \Lambda_{\text{auth}} \ge 0.70 \land \Phi_{\text{disc}} \le -0.50 \land \rho_{\text{AI}} = 0 \\
0, & \text{otherwise}
\end{cases}$$

The final posterior probability for class $k \in \{0, 1, 2, 3\}$ is evaluated via optimal temperature scaling:

$$P^*(y = k \mid x) = \frac{\exp\left( \frac{s_k + g_k}{T^*} \right)}{\sum_{j=0}^3 \exp\left( \frac{s_j + g_j}{T^*} \right)}$$

Where $T^* = 1.55$ is the temperature parameter optimized to minimize Expected Calibration Error (ECE $< 0.05$) *(UNVERIFIED HYPOTHESIS: calibrated on leaky synthetic data)*.

---

### Equation 7: Dynamic Margin Uncertainty Gating Equation ($\mathcal{U}(x)$)

To safeguard writers against false accusations when a document lies directly on an ambiguous classification boundary:

$$\mathcal{U}(x) = 1.0 - \left( P^*(y_{(1)} \mid x) - P^*(y_{(2)} \mid x) \right) \cdot \left( \frac{P^*(y_{(1)} \mid x)}{\tau_{\text{threshold}}} \right)$$

Where $y_{(1)} = \arg\max_k P^*(y = k \mid x)$ and $y_{(2)}$ is the runner-up class.
If $\mathcal{U}(x) \ge \tau_{\text{uncertain}} = 0.70$ *(UNVERIFIED HYPOTHESIS: theoretical threshold)* or $P^*(y_{(1)} \mid x) < \tau_{\text{threshold}}$, the verdict is gated:

$$\text{Final Verdict} = \begin{cases}
\text{"Uncertain (Low Margin)"}, & \text{if } \mathcal{U}(x) \ge 0.70 \\
\arg\max_k P^*(y = k \mid x), & \text{otherwise}
\end{cases}$$

---

## 3. Empirical Verification: Sedaris vs ChatGPT

> [!NOTE]
> **Single-Pair Demonstration (UNVERIFIED HYPOTHESIS)**: The table below reflects outputs on an isolated pair of sample texts. These values are illustrative demonstrations rather than statistically verified benchmarks across representative corpora.

Below are the empirical values produced by these mathematical equations when evaluated on real-world benchmark texts:

| Mathematical Metric | Equation | David Sedaris (*Us and Them*, Human Memoir) | ChatGPT (Essay on AI in Education) | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Authorial Affinity** | $\Lambda_{\text{auth}}$ | **$0.7418$** | **$0.1352$** | **$+0.6066$ Gap** (Clear separation) |
| **Syntactic Burstiness** | $\mathcal{B}_{\text{syntax}}$ | **$0.7068$** | **$0.4677$** | Human cadence is 51% more dynamic |
| **Cadence Delta** | $\Delta_{\text{cadence}}$ | $10.9$ words | $7.5$ words | Erratic human sentence length shifts |
| **Cadence Curvature** | $\kappa_{\text{cadence}}$ | $20.8$ words | $12.3$ words | Human clause acceleration |
| **Discourse Polarity** | $\Phi_{\text{disc}}$ | **$-0.8776$** | **$+0.3464$** | Extreme negative polarity for memoir |
| **AI Marker Rate** | $\rho_{\text{AI}}$ | $0.00\%$ | $0.53\%$ | AI transition cliches present |
| **Human Marker Rate** | $\rho_{\text{Hum}}$ | $7.17\%$ | $0.00\%$ | High first-person narrative voice |
| **Compressibility** | $C_{\text{Deflate}}$ | $0.540$ | $0.433$ | LLM text compresses significantly more |
| **Binoculars Ratio** | $\mathcal{R}_{\text{Binoc}}$ | $12.99$ | $17.59$ | Divergence between entropy and compressibility |
| **Lexical Richness** | $\Omega_{\text{lex}}$ | $0.8243$ | $0.8594$ | Length-invariant vocabulary dispersion |
| **Final Classified Probability** | $P^*(y \mid x)$ | **Human-written: $76.6\%$** | **AI-generated: $91.6\%$** | Single pair demo *(UNVERIFIED HYPOTHESIS)* |

---

## 4. Code Implementation Mapping

| Mathematical Formulation | Python Implementation | Source Location |
| :--- | :--- | :--- |
| $\Omega_{\text{lex}}, \mathcal{B}_{\text{syntax}}, \Phi_{\text{disc}}, \mathcal{R}_{\text{Binoc}}, \Lambda_{\text{auth}}$ | `compute_mathematical_equations()` | `backend/stylometrics.py#L365-L420` |
| Simpson's Index $D_{\text{Simpson}}$ | `compute_lexical_diversity()` | `backend/stylometrics.py#L210-L225` |
| Honoré's Statistic $\tilde{R}_{\text{Honoré}}$ | `compute_lexical_diversity()` | `backend/stylometrics.py#L226-L235` |
| Cadence Delta $\Delta$ & Curvature $\kappa$ | `compute_syntactic_variance()` | `backend/stylometrics.py#L265-L280` |
| Lossless Deflate $C_{\text{Deflate}}$ | `compute_compression_ratio()` | `backend/stylometrics.py#L73-L84` |
| Bayesian Fusion & Guardrail $g_k(x)$ | `_apply_meta_classifier_and_calibration()` | `backend/runtime_engine.py#L180-L215` |
| Temperature Scaling $P^*(y \mid x)$ | `_apply_meta_classifier_and_calibration()` | `backend/runtime_engine.py#L210-L220` |
| Margin Uncertainty Gating $\mathcal{U}(x)$ | `analyze_text()` | `backend/runtime_engine.py#L295-L315` |

---

## 5. Compliance with Project Target Hardware Constraints

1. **Pure ONNX INT8 Execution**: All equations are computed via vectorized NumPy and standard math primitives in $<1.5\text{ms}$ on 2 CPU threads *(Preliminary estimate)*.
2. **Zero PyTorch at Runtime**: No neural tensor operations required during inference.
3. **Total Latency**: Document analysis (500 words) executes in **$0.139\text{s}$** (Target: $\le 15\text{s}$) *(Preliminary single-run measurement)*.
4. **Memory Ceiling**: Peak process RSS is **$178.6\text{MB}$** (Target: $\le 1500\text{MB}$) *(Preliminary single-run measurement)*.
5. **Fairness**: Non-native (ESL) false-positive rate is **$1.1\%$** [0.3–3.9] on verified real learner corpora, meeting the fairness boundary.

---

## 6. Distillation & Teacher Formulations (ZeroGPT Model Integration)

### Equation 8: Calibrated Knowledge Distillation Loss ($\mathcal{L}_{\text{distill}}$)
To distill ZeroGPT's sentence-level token predictability and perplexity dynamics into our compact student transformer (`all-MiniLM-L6-v2`) without inheriting ZeroGPT's false-positive bias on simple human prose:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_{\text{student}}}{T}\right) \,\Big\|\, \sigma\left(\frac{z_{\text{ZeroGPT}}}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y_{\text{true}}, z_{\text{student}})$$

Where:
- $T = 2.0$ is the distillation softening temperature.
- $\alpha = 0.5$ balances teacher mimicry with ground-truth empirical calibration.
- $z_{\text{ZeroGPT}}$ is the calibrated soft target derived from ZeroGPT's sentence token perplexity:
  $$p_{\text{AI}}(s) = \frac{1}{1 + \exp\left(\text{clip}\left(\frac{\text{PPL}(s) - 35.0}{8.0}, -40, 40\right)\right)}$$

### Equation 9: Word-Count Weighted Sentence Coverage ($\text{fakePercentage}$)
ZeroGPT's headline score avoids document-level logit dilution on multi-paragraph texts by aggregating across flagged sentences:

$$\text{fakePercentage} = \frac{\sum_{s \in \mathcal{H}} |w(s)|}{\sum_{s \in \mathcal{S}} |w(s)|} \times 100\%$$

Where $\mathcal{H} = \{s \in \mathcal{S} \mid p_{\text{AI}}(s) \ge \tau_{\text{sentence}}\}$ is the set of flagged AI sentences and $|w(s)|$ is the word count of sentence $s$.

### Measured Empirical Verification (`cand:zerogpt_distilled` on Dev Split):
- **1% FPR Threshold**: Normalized from $0.9804$ to **$0.7101$**
- **Claude Opus 5.5 Raw TPR**: Elevated from $0.0\%$ to **$10.4\%$** [5.4–19.2]
- **Claude Sonnet 5.5 Raw TPR**: Elevated from $0.0\%$ to **$15.4\%$** [9.0–25.0]
- **Claude Sonnet A3 Humanizer TPR**: Elevated from $0.0\%$ to **$30.0\%$** [10.8–60.3]
- **600+ Word Long Document TPR**: Elevated from $0.0\%$ to **$22.2\%$** [9.0–45.2]
- **ESL Learner Fairness**: $1.1\%$ [0.3–3.9] realized FPR ($0.0\%$ on CEFR Bands A & B)

---

## 7. Multi-Teacher Ensemble Knowledge Distillation Formulations

### Equation 10: Fused Multi-Teacher Probability Distribution ($\mathbf{q}_{\text{fused}}$)
To harness complementary signals from disparate detection paradigms—causal language model predictability (ZeroGPT), 4-class paraphrase-sensitive taxonomy (QuillBot), and bidirectional contextual transformer representations (RoBERTa & ModernBERT)—we formulate the fused teacher probability vector $\mathbf{q}_{\text{fused}} \in \Delta^3$:

$$\mathbf{q}_{\text{fused}}(x) = \sum_{k=1}^K w_k(x) \cdot \mathbf{p}_k(x)$$

Where:
- $\mathbf{p}_{\text{ZeroGPT}}(x)$ supplies token-level predictability and inter-sentence burstiness variance shielding $\sigma^2_{\text{PPL}}$.
- $\mathbf{p}_{\text{SeqEns}}(x) = \frac{1}{2}\left[\sigma\left(\mathbf{z}_{\text{RoBERTa}}(x)\right) + \sigma\left(\mathbf{z}_{\text{ModernBERT}}(x)\right)\right]$ supplies sequence-level contextual attention posteriors.
- $\mathbf{p}_{\text{QuillBot}}(x)$ maps the 4-class taxonomy over hybrid refinement and paraphrase modes.
- Adaptive gating weights $w_k(x)$ dynamically downweight causal perplexity on narrative/ESL human prose while upweighting contextual transformers on frontier LLMs (Claude Opus/Sonnet 5.5).

### Equation 11: Multi-Teacher Distillation Objective ($\mathcal{L}_{\text{multi-distill}}$)
The student transformer (`sentence-transformers/all-MiniLM-L6-v2`) minimizes a composite loss over temperature-scaled soft KL divergence and hard ground-truth cross-entropy:

$$\mathcal{L}_{\text{multi-distill}} = \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{\mathbf{z}_s}{T}\right) \;\Big\|\; \sigma\left(\frac{\mathbf{q}_{\text{fused}}}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{z}_s, y_{\text{true}})$$

With hyperparameters selected via empirical grid search:
$$T \in \{1.5, 2.0, 2.5\}, \quad \alpha \in \{0.4, 0.5, 0.6\}$$
Optimizing across this surface transfers nuanced authorial representations into an ultra-compact $\le 25\text{ MB}$ INT8 ONNX edge artifact.

### Measured Empirical Verification (`cand:multi_teacher_distilled` on Dev Split):
- **1% FPR Decision Threshold**: Normalized from $0.9752$ to **$0.6898$**
- **5% FPR Decision Threshold**: $0.6175$
- **AUROC (AI vs Clean Human)**: **$0.7227$** (+0.0717 pts over baseline)
- **ESL Learner Realized FPR**: **$0.5\%$** [0.1–3.0] ($0.0\%$ on CEFR Bands A & B)
- **Claude Opus 5.5 Raw TPR**: **$10.4\%$** [5.4–19.2]
- **Claude Sonnet 5.5 Raw TPR**: **$14.1\%$** [8.1–23.5]
- **Claude Sonnet A3 Humanizer TPR**: **$10.0\%$** [1.8–40.4]
- **MAGE GPT-4 Raw TPR**: **$15.9\%$** [9.7–25.0]
- **RAID Mistral-Chat Raw TPR**: **$75.0\%$** [30.1–95.4]
- **Pooled TPR @1% FPR**: **$12.6\%$** [9.9–15.8] | **@5% FPR**: **$30.1\%$** [26.2–34.3]
- **Quantized Artifact Size**: **21.96 MB** (`student_model_int8.onnx`, 74.7% compression)



