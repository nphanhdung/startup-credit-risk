# Data

The raw dataset is not stored in this repository. It comes from Crunchbase, and redistributing it may not be allowed under Crunchbase's terms.

To run the analysis, download the Crunchbase start-up dataset (about 66,000 companies, last updated 2015) and save it as:

```
data/startup_failures.csv
```

<!-- TODO: add the exact download link (e.g. the Kaggle page) you used. -->
Source: Crunchbase start-up dataset (2015 export)

Expected columns: `permalink`, `name`, `homepage_url`, `category_list`, `funding_total_usd`, `status`, `country_code`, `state_code`, `region`, `city`, `funding_rounds`, `founded_at`, `first_funding_at`, `last_funding_at`.

The model only uses company-level information: dates, funding, location and categories. Company names and URLs are not used as features.
