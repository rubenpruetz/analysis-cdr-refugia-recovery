# analysis-cdr-refugia-recovery
This is the repository for the analysis of warming-related and land use change related impacts on climate refugia

# Overview of project scripts
The project is composed of the following Python scripts:
- **required_functions.py:** This script contains the required project functions.
- **1_data_preprocessing.py:** This script preprocesses the climate refugia data and the scenario-based land use data (from the models AIM, GCAM, GLOBIOM, IMAGE, and REMIND-MAgPIE). This is necessary to ensure a uniform spatial resolution, a uniform Coordinate Reference System, and uniform units. The scripts also includes further preprocessing steps for the subsequent analysis steps, as detailed in the script itself.
- **2_data_analysis.py:** This script is used to calculate warming-related and mitigation-related climate refugia 'loss' at the global and at country level, to identify model consensus, and to compare mitigation-related land allocation within refugia before and after overshoot.
- **3_supplementary_analysis_a.py:** This script contains supplementary code to complement the analysis in the file 2_data_analysis.py.
- **4_supplementary_analysis_b.py:** This script contains supplementary code to complement the analysis in the file 2_data_analysis.py.

# Overview of project lookup tables
The project contains the following lookup tables that are required to run the Python scripts:
- lookup_table_uea_resample_20km.csv
- lookup_table_uea_interpol_20km_2digits.csv
- lookup_table_aim_nc_files.csv
- lookup_table_gcam_nc_files.csv
- lookup_table_globiom_nc_files.csv
- lookup_table_image_nc_files.csv
- lookup_table_image_nc_files_preprocessing.csv
- lookup_table_magpie_nc_files.csv
- lookup_table_ar_bioenergy_files_all_models.csv

These lookup tables are primarily used to import, manage, and export files with different names.

# Install from GitHub
To clone this repository, use the following command:
```
git clone https://github.com/rubenpruetz/analysis-cdr-refugia-recovery.git
cd analysis-cdr-refugia-recovery
```
If Python 3.11 is already installed, package dependencies can be installed within the standard Python environment with the following command:
```
pip install -r requirements.txt
```
You can run the project scripts (specified above) from the terminal with the following command:
```
python scripts/SELECT-FROM-SCRIPTS-ABOVE.py
```
Alternatively, the scripts can be run in various integrated development environment (the project was setup and run in Spyder 6.03). Note: The spatially explicit data required to run the scripts is not provided in this repository as not all required input data is publicly available. See data availability statement below.

# Data availability
The global and country-level data outputs are made available at: https://doi.org/10.5281/zenodo.15497447 
Climate refugia data can be made available upon reasonable request.
Non-spatial scenario data from the AR6 Scenarios Database are available at: https://doi.org/10.5281/zenodo.7197970 
Land use data from AIM-SSP/RCP Ver2018 are available at: http://doi.org/10.18959/20180403.001   
Land use data from GCAM-Demeter are available at: https://doi.org/10.25584/data.2020-07.1357/1644253 
Land use data from GLOBIOM are available at: https://doi.org/10.5281/zenodo.15964077
Land use data from IMAGE 3.0.1 are available at: https://doi.org/10.5281/zenodo.17046335 
Land use data from REMIND-MAgPIE 1.6-3.0 are available at: https://doi.org/10.5281/zenodo.17047534
Data on the constrained reforestation potential map are available at: https://www.naturebase.org  
Data on the constrained biomass plantation map are available at: https://doi.org/10.5281/zenodo.14514051
World administrative boundaries data are available at: https://geonode.wfp.org/layers/geonode%3Awld_bnd_adm0_wfp

# License
**Creative Commons Attribution 4.0 International**.
