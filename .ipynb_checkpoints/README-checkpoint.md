# lensqso-specfit

A Python package for jointly fitting the spectra of multiple gravitationally lensed quasar images, with flexibility for microlensing-induced spectral deformations.

See [filename](example_usage/Example_notebook.ipynb) for a demonstration on a simple example.

## Features

- **Joint multi-spectrum fitting** — models several quasar image spectra simultaneously, with the option to share spectral feature parameters across images or to fit them separately in order to allow for microlensing deformations.
- **Variety of spectral components** — supports power-law and polynomial continuuum, single emission lines (Gauss-Hermite and Voigt profiles), narrow doublets, and emission line templates (optical FeII, and NIR FeII+MgII).
- **Efficient linear/non-linear parameter separation** — amplitude-like (linear) parameters are solved analytically at each likelihood evaluation via weighted least squares, reducing the dimensionality of the non-linear optimization.
- **Multiple optimizers** — bound derivative-free optimization (`COBYQA`), Particle Swarm Optimization (via `lenstronomy`), and MCMC sampling (via `emcee`), composable into a `FittingSequence`.
- **Flux ratio posteriors** — derives posterior distributions on image flux ratios for any modeled emission feature directly from MCMC chains.

## Repository structure

```
lensqso-specfit/
├── spectrum_model.py        # QuasarSpectrum and LinearParamHandler classes
├── multi_spectrum.py        # MultiSpectrum class (joint model for N images)
├── multi_spec_fitter.py     # Optimizer, PSO, MCMCSampler, FittingSequence
├── QSO_spectral_features.py # Spectral component definitions (lines, continua, templates)
├── FeII_template_4000_5500/ # Iron emission template data
└── example_usage/           # Example spectra and Jupyter notebook showcasing the different features
```

## Acknowledgements
The current architecture of lensqso-specfit was developed in Paugnat et al. (2026). Please cite this paper if you use the package in your research.

