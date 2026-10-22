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

## Results (original course submission)

- **AUC ≈ 0.78** (Gini ≈ 0.56) on held-out data, compared with 0.50 for random guessing.
- **Calibration** lowered the Brier score from 0.063 to 0.062. A naive model scores about 0.07.
- **Strongest predictors:** time since the last funding round, business age and total funding.
- **Risk buckets separate risk clearly.** The observed failure rate rises from about 2% in bucket A to about 83% in bucket F.
- **Expected loss** across the eligible portfolio is about 4% of the amount lent.

The model rests on public funding data only, with no financial statements or repayment history, so it shows the method rather than giving a production-ready credit model.

> The numbers above come from the original course analysis. This version fixes several issues (see [Changes](#changes-from-the-course-version)), so a fresh run will give slightly different figures. Run `python -m startup_credit_risk.pipeline` to reproduce them.

## How to run

```bash
git clone https://github.com/<your-username>/startup-credit-risk.git
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
- **Honest validation.** Bucket failure rates are measured on the untouched test set, not on all the data.
- **Reproducibility.** Time-based features are measured up to the last date in the data, not today's date, so results no longer change from day to day.
- **Missing categories.** Companies with no category are now labelled "Other". Before, they were put in "Administrative Services" by mistake.
- **Class weight.** It is computed from the training data instead of being hard-coded.
- **Code structure.** The notebook code was moved into tested modules, with all settings in one config file.

## Data

The raw data is **not included** in this repository. See [data/README.md](data/README.md) for where to get it. It describes companies, not individuals.

## Tools

Python, pandas, scikit-learn, matplotlib, pytest
