# IAU - Exploratory Data Analysis

Exploratory analysis of an IoT/APU measurement dataset, done as the first phase of
the Artificial Intelligence course project at FIIT STU.

## Dataset

`174/` contains three related files with different separators:

| File | Separator | Rows |
|---|---|---:|
| `readings.csv` | tab | 12,288 |
| `units.csv` | comma | 2,000 |
| `gateways.csv` | semicolon | 16,000 |

Each row in `readings.csv` is one measurement of an APU unit (14 attributes).

## What the notebook covers

- Loading three CSV files with different separators.
- Structural profiling: shape, dtypes, missing values, duplicates, cardinality.
- Summary statistics and distributions.
- Relationships between readings, units and gateways.
- Data-quality observations and their impact on later modelling.

Open `phase1.ipynb` in Jupyter; the dataset is included, so it runs offline.

```bash
pip install jupyter pandas numpy matplotlib seaborn scipy
jupyter notebook phase1.ipynb
```

## Authors

Group project:

- **Ondrej Nebus** - 66 % (sections 1.2, 1.3)
- Marcel Novomesky - 34 % (section 1.1 A-E)

FIIT STU, 2025/26.
