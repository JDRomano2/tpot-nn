
## Repository structure 

### Datasets
- 300 simulated datasets can be found in the [`data_300`](data_300) folder.
Details on data simulation are provided at [https://github.com/lelaboratoire/rethink-prs/edit/master/README.md](https://github.com/lelaboratoire/rethink-prs/edit/master/README.md).

- QSAR datasets (`.csv`) are located in the [`qsar`](qsar) folder.
NPDR-filtered QSAR datasets are in [`qsar_npdr`](qsar_npdr) folder.

- [`subsets`](subsets) contain smaller QSAR datasets for testing purposes.

### Literature review
- [`existing_performance`](existing_performance) summarizes existing accuracies on the QSAR datasets of previous studies (see [`papers`](papers)).

### Results
- [`pipelines`](pipelines) and [`accuracies`](accuracies) save the preliminary results of running TPOT-MLP on the QSAR datasets.
- [`predictions`](predictions) are predicted outcome on the testing set.