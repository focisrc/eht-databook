# The Event Horizon Telescope Data Processing Book

A [Jupyter Book](https://jupyterbook.org/) for
[Event Horizon Telescope (EHT)](https://eventhorizontelescope.org/)
data processing.
It covers topics from understanding the foundation of Very Long
Baseline Interferometry (VLBI) to actual data processing methods and
practices.

## Contributing

All materials in this Jupyter Book are stored a special flavour of
Markdown called
[MyST (or Markedly Structured Text)](https://myst-parser.readthedocs.io/).

To edit the materials locally as Jupyter Notebooks, simply use
Jupytext to create a pair notebook:
```
jupytext-3.10 --sync --to ipynb [chapter].md
```
All changed made to the notebook will be automatically synced to the MyST
markdown file.
