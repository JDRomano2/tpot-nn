### Where to start
The METAB dataset example: code for training TPOT to predict the outcome of  is in [`real-data-train-metab.ipynb`](real-data-train-metab.ipynb). This file takes a while to run, so please change `n_gen` and `n_pop` to smaller numbers if you want to test. This code finds the best pipeline and writes it as `*_METAB_*.py` in [`pipelines`](pipelines). [`real-data-test-metab.ipynb`](real-data-test-metab.ipynb) reads in this pipeline and produce predictions (and corresponding accuracy) on the testing set.

### Datasets
- QSAR datasets (`.csv`) are located in the [`qsar`](qsar) folder.
NPDR-filtered QSAR datasets are in [`qsar_npdr`](qsar_npdr) folder.

- [`subsets`](subsets) contain smaller QSAR datasets for testing purposes.