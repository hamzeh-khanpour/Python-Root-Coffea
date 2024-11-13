# Photon Luminosity Spectrum and Tau-Tau Production Cross-Section Calculation

## Overview

This repository contains Python scripts for calculating the **elastic/inelastic photon-photon luminosity spectrum** (`S_yy`) and 
the **tau-tau production cross-section** for the process \( ep \rightarrow e (\gamma\gamma \rightarrow \tau^+\tau^-) p^{(*)} \). 

## Features

- **Photon-Photon Luminosity Spectrum Calculation** (`S_yy`):
  - Computes `S_yy` for a range of center-of-mass energies `W`.
  - Allows flexibility for analyzing dimuon or Higgsinos production cross-sections as well. 
  
- **Tau-Tau Production Cross-Section Calculation**:
  - Calculates the production cross-section for tau pairs at specified W values using the interpolated `S_yy`.
  - Integrates cross-section values from a user-specified threshold `W_0` up to the center-of-mass energy.
  - Includes options for visualizing results with Matplotlib.

## File Structure

### Key Files

- **`flux_el_yy_atW(W, eEbeam, pEbeam)`**  
  - Computes the `S_yy` values for inelastic photon-photon interactions.
  - Includes options for saving and plotting results, with customized parameters for MN, \( Q^2 \) max values, and W range.

- **`cs_tautau_w(W)`**  
  - Calculates and integrates the tau-tau production cross-section using `S_yy` data from a precomputed file.
  - Provides options to skip integration for W values where `S_yy` is zero, improving performance and accuracy.

- **`Integrated_elastic_tau_tau_cross_section_final_version_using_vegas.py`**   and **`Integrated_inelastic_tau_tau_cross_section_final_version_using_vegas.py`**
  - Compares `S_yy` values between a simple approximation and the corrected inelastic model.
  - Computes and plots the relative difference between the two models for a comprehensive comparison.

- **Data Files**:  
  - **`Inelastic_Photon_Luminosity_Spectrum_MNmax_<value>_q2emax_<value>_q2pmax_<value>_using_vegas.txt`**  
    - Precomputed `S_yy` values for inelastic photon-photon interactions, generated using the phthon script.

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`

Install dependencies via:
```bash
pip install numpy scipy matplotlib
```

## Usage

### 1. Generate Photon-Photon Luminosity Spectrum (`S_yy` and cross-sections calculations) - elastic case

Run `Integrated_elastic_tau_tau_cross_section_final_version_using_vegas.py` to compute `S_yy` and integrated tau-tau production cross-section for elastic interactions. 
Customize parameters such as the beam energies, \( Q^2 \) maximum values, and MN upper limit before running the script.

```bash
python Integrated_elastic_tau_tau_cross_section_final_version_using_vegas.py
```

Results are saved in `Inelastic_Photon_Luminosity_Spectrum_MNmax_<value>_q2emax_<value>_q2pmax_<value>_using_vegas.txt`.

### 2. Generate Photon-Photon Luminosity Spectrum (`S_yy` and cross-sections calculations)  - inelastic case

Using `Integrated_inelastic_tau_tau_cross_section_final_version_using_vegas.py`, calculate the iSyy and integrated tau-tau production cross-section. 
This script reads `S_yy` data, interpolates it, and performs integration to compute the cross-section at each threshold energy `W0`.

```bash
python Integrated_inelastic_tau_tau_cross_section_final_version_using_vegas.py
```

The script outputs the integrated tau-tau cross-section and saves the result plot.

## Results and Visualization

Each script outputs results in both text and graphical formats:
- `S_yy` and cross-section data are saved in `.txt` files.
- Plots are generated for the luminosity spectrum, tau-tau cross-section, and model comparisons. Files are saved as `.pdf` and `.jpg`.

