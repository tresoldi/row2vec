# Benchmarks

What each mode costs, and what it keeps, as the table grows. Every size is a
random sample of the bundled [Adult census data](https://github.com/tresoldi/row2vec/tree/main/data)
(14 mixed numeric and categorical columns, income as the target).
`compare_modes` fits each mode on 75% of the rows and scores it on the other
25%, so quality is always measured on rows the embedding never saw.

Reproduce, or run it on your own machine:

```bash
python scripts/benchmarks.py            # full run, takes several minutes, mostly t-SNE
python scripts/benchmarks.py --quick    # two small sizes
```

Timings depend on the hardware: compare the modes with each other, not with
your laptop. Memory is not reported, because TensorFlow allocates outside
anything Python can measure.

Run 2026-10-02 on Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41, Python 3.12.13, row2vec 0.4.0. Embedding dimension 4, 20 epochs for neural modes.

### Fit time (seconds)

| mode | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| baseline | - | - | - | - | - |
| contrastive | 3.002 | 4.678 | 9.875 | 16.736 | 25.983 |
| pca | 0.058 | 0.059 | 0.109 | 0.169 | 0.126 |
| target | 2.274 | 2.632 | 3.01 | 3.96 | 3.889 |
| tsne | 7.695 | 24.584 | 51.56 | 47.029 | 51.053 |
| umap | 0.531 | 1.04 | 2.503 | 8.697 | 20.726 |
| unsupervised | 2.224 | 2.813 | 3.335 | 3.819 | 5.117 |

### Trustworthiness (held-out rows)

| mode | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| baseline | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| contrastive | 0.895 | 0.893 | 0.913 | 0.936 | 0.943 |
| pca | 0.902 | 0.895 | 0.895 | 0.926 | 0.929 |
| target | 0.781 | 0.734 | 0.758 | 0.784 | 0.78 |
| tsne | 0.955 | 0.974 | 0.983 | 0.984 | 0.982 |
| umap | 0.898 | 0.922 | 0.941 | 0.949 | 0.956 |
| unsupervised | 0.855 | 0.904 | 0.931 | 0.948 | 0.96 |

### k-NN accuracy on the embedding (held-out rows)

| mode | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| baseline | 0.824 | 0.808 | 0.812 | 0.825 | 0.826 |
| contrastive | 0.76 | 0.78 | 0.784 | 0.797 | 0.806 |
| pca | 0.8 | 0.788 | 0.806 | 0.823 | 0.814 |
| target | 0.784 | 0.828 | 0.796 | 0.846 | 0.834 |
| tsne | - | - | - | - | - |
| umap | 0.8 | 0.772 | 0.79 | 0.836 | 0.803 |
| unsupervised | 0.76 | 0.784 | 0.808 | 0.817 | 0.822 |

t-SNE is scored on at most 1500 rows (it cannot embed unseen rows, and its cost grows quickly), so its time stops growing past that size.

## Reading the tables

- **PCA is nearly free** and, on this data, keeps neighbourhoods and downstream
  accuracy within a few points of the heavier methods. Start there.
- **The autoencoder (`unsupervised`) improves with more rows**: its
  trustworthiness climbs from 0.86 at 500 rows to 0.96 at 8000, overtaking PCA
  from about 2000. Its time grows slowly, because training time is dominated by
  epochs, not rows.
- **`target` wins on the downstream score** (it is trained for the label) and
  loses on trustworthiness. That is the trade it makes: use it when the
  embedding feeds a classifier, not when you want to see the table's geometry.
- **`contrastive` costs the most of the neural modes** here, since
  `auto_pairs="neighbors"` builds and trains on pairs; its quality tracks the
  autoencoder's.
- **UMAP's time grows faster than the row count** in this range (0.5s at 500
  rows, 21s at 8000), while PCA and the autoencoder barely move.
- **t-SNE is the most faithful and by far the slowest**, and cannot embed rows
  it was not fitted on, so it has no downstream score. Use it for a picture,
  not a pipeline.
- **The baseline** (the preprocessed features, no embedding) sets the ceiling
  for the downstream score: a 4-dimensional embedding rarely beats the full
  encoded table, and the point is how much it keeps for the size.
