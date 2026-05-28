# HGBO-DSE Documentation

This directory keeps documentation and curated experiment evidence out of the
source root.

- `setup/`: environment, Docker, and host execution notes.
- `reports/`: maintained reproduction reports and longer diagnostic histories.
- `experiments/`: stable JSON summaries referenced by the thesis/reporting
  narrative. Short smoke-run outputs are intentionally not kept here.

Generated training runs should continue to write under `img/training/` or an
external work directory. Do not add raw datasets, architecture caches, virtual
environments, or one-off smoke summaries to this docs tree.
