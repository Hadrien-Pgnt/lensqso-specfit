
<img align="right" height="240" src="https://github.com/user-attachments/assets/0edc286e-a0b2-47e9-b233-64b23a633980"> 

<h3><strong>$\Huge \texttt{lensqso-specfit}$</strong></h3>  

A Python package for the joint spectral modeling of multiple gravitationally lensed quasar images, with flexibility to allow for microlensing deformations.

See [example_usage/Example_notebook.ipynb](example_usage/Example_notebook.ipynb) for a demonstration on a simple example.



## Features

- **Joint multi-spectrum fitting** — models several quasar image spectra simultaneously, with the option to share spectral feature parameters across images or to fit them separately in order to allow for microlensing deformations.
- **Variety of spectral components** — supports power-law and polynomial continuuum, single emission lines (Gauss-Hermite and Voigt profiles), narrow doublets, and emission line templates (optical FeII, and NIR FeII+MgII).
- **Efficient linear/non-linear parameter separation** — amplitude-like (linear) parameters are solved analytically at each likelihood evaluation via weighted least squares, reducing the dimensionality of the non-linear optimization.
- **Multiple optimizers** — bound derivative-free optimization (`COBYQA`), Particle Swarm Optimization (via `lenstronomy`), and MCMC sampling (via `emcee`), composable into a `FittingSequence`.
- **Flux ratio posteriors** — derives posterior distributions on image flux ratios for any modeled emission feature directly from MCMC chains.

## Repository structure

```
lensqso-specfit/
├── QSO_spectral_features.py # Spectral component definitions (line profiles, doublet, continuum, templates)
├── spectrum_model.py        # QuasarSpectrum class to represent a single spectrum (with LinearParamHandler)
├── multi_spectrum.py        # MultiSpectrum class to jointly represent multiple spectra 
├── multi_spec_fitter.py     # FittingSequence and optimizers/MCMC sampler
├── FeII_template_4000_5500/ # Optical Fe II emission template data
├── FeII+MgII_template_8300_11600.txt # NIR FeII+MgII emission template
└── example_usage/           # Example spectra and Jupyter notebook showcasing the different features
```

## Acknowledgements
The current architecture of lensqso-specfit was developed in Paugnat et al. (2026). Please cite this paper if you use the package in your research.

