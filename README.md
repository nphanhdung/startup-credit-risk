# Start-up Credit Risk Scoring

Can a lender tell, from public funding data alone, which start-ups are likely to fail and what loan each should be offered?

This project builds a **calibrated failure-probability model** for start-ups, turns it into a **0–1000 credit score** with six risk buckets, and designs a **loan product for each bucket**: lending amount, term and risk premium.

It began as a group project for the *Statistical Consulting* course at KU Leuven (2025). This repository is a cleaned-up, reproducible version of the analysis code.

## Approach

```
Raw Crunchbase data ──► Cleaning & features ──► Random forest ──► Isotonic calibration
                                                                          │
           Loan product per bucket ◄── Risk buckets A–F ◄── Credit score 0–1000
```

1. **Data and scope.** The data covers about 66,000 companies from Crunchbase (last updated 2015). The analysis keeps start-ups founded in 2005 or later, in countries that hold more than 1% of the companies. A company counts as failed ("default") when its status is `closed`.
2. **Features.**
   - Total funding and number of funding rounds.
   - Business age, time between founding and first funding, time between first and last funding, and time since last funding.
   - Founding and funding years, number of categories.
   - Country, state, and one of 47 industry groups mapped from the raw Crunchbase categories.
3. **Model.**
   - A random forest with class weights handles the imbalance: about 1 failure for every 12 surviving companies.
   - The data is split 60/20/20 into training, calibration and test sets.
   - The forest is trained on the training set. Its probabilities are then calibrated with isotonic regression on the separate calibration set, so the scores can be read as real failure probabilities.
4. **Rating scale.** Score = (1 − failure probability) × 1000. The scores are cut into six buckets that mirror the six credit quality steps in EU Regulation 2016/1799.
5. **Loan products.**
   - Buckets A–D get a loan sized as a share of the company's total funding, with a term and a risk premium that grow with risk.
   - Buckets E and F are not eligible.
   - Expected loss = failure probability × loan amount.

| Bucket | Score | Loan (% of funding) | Term | Risk premium |
|---|---|---|---|---|
| A | 950–1000 | 80% | 3–5 years | +3% |
| B | 900–950 | 60% | 3 years | +8% |
| C | 850–900 | 40% | 2 years | +15% |
| D | 750–850 | 25% | 18–24 months | +20% |
| E, F | below 750 | not eligible | – | – |

## Results

All numbers are measured on a **held-out test set** of about 8,100 start-ups that the model never saw during training or calibration. The full scope is 40,473 start-ups, with a 7.3% failure rate.

**Model performance**

| Metric | Uncalibrated | Calibrated | Benchmark |
|---|---|---|---|
| AUC | 0.761 | 0.757 | 0.50 (random guessing) |
| Gini | 0.52 | 0.51 | 0 |
| Brier score | 0.0636 | **0.0626** | 0.0676 (always predict the average) |

Calibration improves the Brier score. That means the predicted probabilities are closer to the real failure rates, which matters more for pricing loans than ranking alone.

**Risk buckets: predicted vs. observed failure rate (test set)**

| Bucket | Start-ups | Avg. score | Predicted failure | Observed failure |
|---|---|---|---|---|
| A | 5,069 | 973 | 2.7% | 2.9% |
| B | 817 | 928 | 7.2% | 8.3% |
| C | 884 | 876 | 12.4% | 11.3% |
| D | 922 | 825 | 17.5% | 17.4% |
| E | 318 | 741 | 26.0% | 28.0% |
| F | 85 | 622 | 37.8% | 30.6% |

In buckets A–E, the observed failure rates follow the predictions closely, so the scale can be used to price risk. Bucket F is small (85 companies), so its rate is uncertain.

**Strongest predictors:** time since the last funding round, business age, total funding, and time from founding to first funding. Start-ups that have not raised money for a long time are the most likely to close.

**Loan portfolio (all eligible start-ups)**

| Bucket | Start-ups | Lending (USD bn) | Expected loss | Risk premium |
|---|---|---|---|---|
| A | 29,374 | 313.5 | 2.5% | 3% |
| B | 3,281 | 17.9 | 7.3% | 8% |
| C | 3,025 | 9.4 | 12.3% | 15% |
| D | 2,204 | 3.3 | 17.1% | 20% |
| **Total** | **37,884** | **344.1** | **3.2%** | |

In every bucket the risk premium covers the expected loss. About 91% of the lending goes to the lowest-risk bucket A.

**Limitations.** The model uses public funding data only, with no financial statements or repayment history. "Closed" is used in place of an actual loan default. The data ends in 2015. The project shows the method; it is not a production-ready credit model.

## How to run

```bash
git clone https://github.com/nphanhdung/startup-credit-risk.git
cd startup-credit-risk
pip install -e ".[dev]"

# 1. Download the data (see data/README.md) to data/startup_failures.csv
# 2. Run the full analysis
python -m startup_credit_risk.pipeline --data data/startup_failures.csv --out outputs

# Run the tests (use synthetic data, no download needed)
pytest
```

The pipeline writes these files to `outputs/`:

- `metrics.json`: AUC, Gini and Brier scores
- `buckets_test_set.csv`: predicted vs. observed failure rate per bucket
- `portfolio.csv`: lending and expected loss per bucket
- `feature_importance.csv`
- figures: calibration curve, ROC, feature importance, score distribution and the industry risk map

## Project structure

```
src/startup_credit_risk/
  config.py         all settings: scope, split sizes, bucket ranges, loan terms
  industries.py     Crunchbase category -> industry group mapping
  preprocessing.py  cleaning, date fixes, feature engineering, scope filters
  model.py          train / calibrate / evaluate
  scoring.py        credit score and risk buckets
  products.py       loan products and portfolio view
  plots.py          figures
  pipeline.py       runs everything end to end
notebooks/          walkthrough notebook
tests/              unit and end-to-end tests on synthetic data
```

## Changes from the course version

- **Calibration.** The forest is now trained on the training set and calibrated on a separate calibration set. Before, both steps used only the 20% calibration set.
- **Honest validation.** Bucket failure rates are measured on the untouched test set, not on all the data. The course version reported 83% observed failure in bucket F, but that was measured partly on training data; on new data it is about 31%. Its AUC of 0.78 dropped to 0.76 for the same reason.
- **Reproducibility.** Time-based features are measured up to the last date in the data, not today's date, so results no longer change from day to day.
- **Missing categories.** Companies with no category are now labelled "Other". Before, they were put in "Administrative Services" by mistake.
- **Class weight.** It is computed from the training data instead of being hard-coded.
- **Code structure.** The notebook code was moved into tested modules, with all settings in one config file.

## Data

The raw data is **not included** in this repository. See [data/README.md](data/README.md) for where to get it. It describes companies, not individuals.

## Tools

Python, pandas, scikit-learn, matplotlib, pytest
