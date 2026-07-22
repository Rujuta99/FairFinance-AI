# FairFinance AI

**Auditing Socioeconomic Bias in LLM-Generated Financial Advice**

[![Paper](https://img.shields.io/badge/paper-PDF-red)](paper/FairFinanceAI_Paper.pdf)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A controlled audit of whether large language models give less complete financial advice to users who signal low socioeconomic status, compared to users who signal high socioeconomic status, with everything else held equal.

---

## TL;DR

We asked GPT-4o-mini, LLaMA-3.3 70B, and Gemini-3 Flash the same six financial questions (retirement, credit debt, investing, emergency fund, insurance, tax) under personas anchored to real Chicago neighborhoods, varying only the user's apparent socioeconomic status. Across 648 responses, all three models gave systematically more complete, wealth-oriented advice to high-SES personas.

| Model | Accuracy Gap (ΔA) | p-value | Effect Size |
|---|---|---|---|
| GPT-4o-mini | +0.048 | 0.015 | small (d=0.34) |
| LLaMA-3.3 70B | +0.133 | <0.001 | medium (d=0.75) |
| Gemini-3 Flash | +0.124 | <0.001 | medium (d=0.56) |

All three models fail the 80% Disparate Impact Rule. Counterfactual ablation shows the gap is driven specifically by **education level**, not neighborhood or job title.

---

## Key Findings

**1. The bias is real and consistent.** All three models, independently trained by different companies, show the same directional bias: high-SES personas get more complete financial advice.

**2. Education is the causal driver.** We isolated each socioeconomic signal (neighborhood, job title, education) by holding the other two neutral. Only education produced a significant gap, in all three models simultaneously (p<0.01). Neighborhood and job title alone produced no significant effect.

**3. Regex-only evaluation hides bias.** Under simple keyword matching, Gemini showed *no* measurable bias (p=0.28). Under semantic NLI-based scoring, the same model showed a highly significant gap (p<0.001). The model was using different vocabulary registers for different perceived audiences, and naive evaluation missed it entirely.

**4. Specific knowledge is being withheld.** Diversified investing is omitted from 80% of low-SES responses vs. 11% of high-SES responses. Capital gains taxation is omitted from 74% of low-SES responses vs. 6% of high-SES responses. Meanwhile, EITC tax credit information and starter emergency fund goals appear *more* often in low-SES responses, suggesting the models tailor advice type rather than uniformly degrading quality.

**5. Retirement advice shows the largest disparity.** Cohen's d ranges from 1.20 to 1.83 across all three models, a large effect by any standard, in a domain where small differences compound into six-figure outcomes over decades.

**6. Fairness instructions are not a universal fix.** Explicitly telling models to treat all users equally narrowed the gap for LLaMA but widened it for GPT, demonstrating that mitigation must be model-specific.

---

## Repository Structure

```
fairfinance-ai/
├── paper/                   Final NeurIPS-format paper (PDF + LaTeX source)
├── poster/                  Conference-style poster (PDF)
├── docs/                    Extended defense / methodology guide
├── src/
│   ├── collect_openai.py        Data collection — GPT-4o-mini
│   ├── collect_groq.py          Data collection — LLaMA-3.3 70B
│   ├── collect_gemini.py        Data collection — Gemini-3 Flash
│   ├── 01_hybrid_scoring.py     Algorithm 1: hybrid regex + NLI accuracy scoring
│   ├── 02_ablation.py           Algorithm 2: counterfactual signal ablation
│   ├── 03_analysis.py           Per-domain gaps, Cohen's d, concept coverage
│   ├── 04_final_metrics.py      Cliff's delta, concept omission, DPD/DIR
│   ├── 05_generate_plots.py     All visualization generation
│   └── 06_ses_classifier.py     SES recoverability classifier (diagnostic)
├── results/
│   ├── plots/                19 generated figures
│   └── csv/                  All numeric results as CSV
└── README.md
```

---

## Methodology

### Dataset

648 responses spanning 12 Chicago neighborhoods (6 high-SES, 6 low-SES, anchored to the [Chicago Community Area Hardship Index](https://www.cmap.illinois.gov/data/community-snapshots)), 3 models, 3 prompt conditions (zero-shot, few-shot, fairness-instructed), and 6 financial domains.

Each persona combines a neighborhood, a job title, and an education level. SES signals co-vary by design in the main dataset to reflect realistic user descriptions; a separate ablation experiment isolates each signal independently.

### Algorithm 1 — Hybrid Accuracy Scoring

A three-layer pipeline scores each response's coverage of a domain-expert rubric (36 concepts across 6 domains):

1. **Regex matching** — fast, high-precision detection of standard terminology
2. **NLI semantic fallback** (`cross-encoder/nli-deberta-v3-small`) — catches paraphrases regex misses, framed as zero-shot entailment classification
3. **Specificity scoring** — a second NLI pass measuring whether covered concepts are stated concretely (named funds, specific limits) or vaguely

Validated against independent human annotation at **Cohen's κ = 0.977** (150 concept-level labels).

### Algorithm 2 — Counterfactual Signal Ablation

The main dataset cannot tell you *which* SES signal causes the gap, since neighborhood, job, and education all vary together. We run a second experiment holding two signals neutral and varying one at a time:

| Condition | Result |
|---|---|
| Neighborhood only | No significant effect (any model) |
| Job title only | No significant effect (any model) |
| Education only | Significant in **all three models** (p<0.01) |

### Evaluation Metrics

- Accuracy gap (ΔA) with 95% bootstrap confidence intervals and Mann-Whitney U tests
- Cliff's delta (nonparametric effect size, preferred over Cohen's d for bounded non-normal data)
- Demographic Parity Difference / Disparate Impact Ratio, evaluated against the EEOC 80% rule
- Concept-level omission rate, identifying exactly which financial concepts are withheld by SES group

---

## Reproducing the Results

```bash
git clone https://github.com/<your-username>/fairfinance-ai.git
cd fairfinance-ai
pip install -r requirements.txt
```

Data collection requires API keys for OpenAI, Groq, and Google AI Studio (set as environment variables; see each `collect_*.py` script).

```bash
# 1. Collect data (requires API keys)
python src/collect_openai.py
python src/collect_groq.py
python src/collect_gemini.py

# 2. Score responses (hybrid regex + NLI, run on GPU recommended)
python src/01_hybrid_scoring.py

# 3. Run the counterfactual ablation experiment
python src/02_ablation.py

# 4. Compute all analysis and fairness metrics
python src/03_analysis.py
python src/04_final_metrics.py

# 5. Generate all plots
python src/05_generate_plots.py
```

Pre-computed results (CSVs and plots) are already included in `results/` if you just want to inspect the numbers without re-running the pipeline.

---

## Paper

The full writeup, including related work, formal problem definition, complete results, and limitations, is available as a [PDF](paper/FairFinanceAI_Paper.pdf) (NeurIPS format) or [LaTeX source](paper/FairFinanceAI_Paper.tex).

---

## Limitations

- The scoring rubric applies universal concept weights and cannot fully distinguish appropriate contextual tailoring (e.g., emphasizing EITC for lower-income users) from genuine knowledge withholding. Reported gaps should be read as a conservative lower bound.
- All personas are synthetic; real user phrasing may interact with model behavior differently.
- Profiles are anchored to Chicago-specific neighborhoods and U.S. financial concepts; generalizability to other geographies or demographic intersections is untested.
- Per-domain sample sizes (n=12 per group per model) limit statistical power for domain-level claims, which should be read as exploratory.

Full discussion in the paper.

---

## Citation

```bibtex
@misc{fairfinanceai2025,
  title  = {FairFinance AI: Auditing Socioeconomic Bias in LLM-Generated Financial Advice},
  author = {Tambewagh, Rujuta and Babaria, Jaimin},
  year   = {2025},
  note   = {Course project, Socially Responsible AI, University of Illinois Chicago}
}
```

---

## Authors

[**Rujuta Tambewagh**](https://github.com/Rujuta99) · [**Jaimin Babaria**](https://github.com/jaiminbabaria)

Department of Computer Science, University of Illinois Chicago

Built for the *Socially Responsible AI* course, with guidance from Professor Lu Cheng.

## License

MIT — see [LICENSE](LICENSE) for details. Dataset and scoring rubric are original work; cite if reused.
