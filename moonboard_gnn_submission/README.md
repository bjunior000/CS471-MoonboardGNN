# MoonBoard Edge-Aware GAT

This folder contains the notebook and dataset for reproducing our MoonBoard difficulty prediction result.

## Files

- `edge_aware_gat_reproducible_experiment.ipynb`
  - Main submission notebook.
  - Run this file in Google Colab.
- `moonGen_scrape_2016_final`
  - Required MoonBoard 2016 dataset file.
  - Upload this file in the dataset upload cell.

## Colab Setting

We used Google Colab free version with a T4 GPU.

The notebook uses `device="auto"`, so it will use CUDA if Colab provides a GPU.

## How to Run

1. Upload `edge_aware_gat_reproducible_experiment.ipynb` to Colab.
2. When the dataset cell runs, it will ask you to upload `moonGen_scrape_2016_final`.
3. Run all cells from top to bottom.

The notebook already includes:

```python
!pip install -q torch scikit-learn scipy pandas tqdm
```

No extra installation is needed.

## Where to See the Result

After the cell below is executed:

```python
results_df, best_result = run_edge_aware_gat_only(CONFIG)
print(json.dumps(best_result, indent=2, ensure_ascii=False))
```

check:

- the printed `edge-aware GAT leaderboard`
- the printed `best_result`
- the saved output files under `runs_edge_aware_gat_only/`

Saved files:

```text
runs_edge_aware_gat_only/
  edge_aware_gat_only_leaderboard.csv
  edge_aware_gat_only_best_summary.json
  config.json
```

## Reported Poster Result

The Edge-Aware GAT numbers used in the poster's Results and Discussion section refer to:

```text
model: edge_gat_seed471
selected by: validation exact accuracy
best epoch: 13
```

Metrics from the saved Colab run:

```text
Exact accuracy:   0.5085
Within-1 accuracy: 0.8514
Macro-F1:         0.3038
```

The notebook trains three seeds by default:

```text
edge_gat_seed471
edge_gat_seed572
edge_gat_seed673
```

and selects the best one by validation exact accuracy.

## Baseline Sources

The MoonBoardRNN works are used as dataset/baseline sources. It is report or preprint, not formally published conference/journal papers(arXiv).

- Yi-Shiou Duh and Ray Chang, "Recurrent Neural Network for MoonBoard Climbing Route Classification and Generation", 2021;