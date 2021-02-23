# Using rooftop photovoltaic generation to cover individual electric vehicle demand - a detailed case study

This repository contains the code of the (still unpublished) paper _Using rooftop photovoltaic generation to cover individual electric vehicle demand - a detailed case study_. The goal of the project is to analyze how much of our mobility energy demand (created by battery electric vehicles) we can cover by using only PV installed on our own home. 

All source code is organized in the `src` directory. 

__Preprocessing:__

- BEV data is preprocessed using the main function in in `src/ecar_data_preprocessing.py.`
- PV generation data is created using scripts in the `src/pv_generation` folder. 

__Experiment:__

- The main experiment is is run using `src/calculate_scenarios.py`.
- The different charging scenarios are defined in `src/methods/scenarios_for_users.py`  

__Evaluation:__ 

- All indicators are calculated using `src\calculate_all_indicators.py`. 
- All figures are created using the scrips in `src\plotting\*`
- All figures are available in `plots`
- `calculate_all_indicators.py` is used to calculate indicators used in the paper (such as the total distance driven),  `generate_all_plots.py` generates all plots.

__Data: __

The BEV data used is part of the [SBB Green Class E-Car pilot study](https://www.sbb.ch/en/travelcards-and-tickets/railpasses/greenclass/about-sbb-green-class/pilot-projects.html) and the related [research project](https://www.research-collection.ethz.ch/handle/20.500.11850/353337). Data access was provided by the [Swiss Federal Railways](https://www.sbb.ch/en/home.html) and can not be published to preserve the privacy of the study participants.










