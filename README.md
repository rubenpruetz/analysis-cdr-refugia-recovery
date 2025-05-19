# analysis-cdr-refugia-recovery
This is the repository for the analysis of warming-related and land use change related impacts on climate refugia

# Overview of project scripts
The project is composed of the following Python scripts:
- **required_functions.py:** This script contains the required project functions.
- **1_data_preprocessing.py:** This script preprocesses the climate refugia data and the scenario-based land use data (from the models AIM, GCAM, GLOBIOM, and IMAGE). This is necessary to ensure a uniform spatial resolution, a uniform Coordinate Reference System, and uniform units. The scripts also includes further preprocessing steps for the subsequent analysis steps, as detailed in the script itself.
- **2_data_analysis.py:** This script is used to calculate warming-related and mitigation-related climate refugia 'loss' at the global and at country level, to identify model consensus, and to compare mitigation-related land allocation within refugia before and after overshoot.

# Overview of project lookup tables
The project contains the following lookup tables that are required to run the Python scripts:
- lookup_table_uea_resample_20km.csv
- lookup_table_uea_interpol_20km.csv
- lookup_table_ssp-rcp_names.csv:
- lookup_table_aim_nc_files.csv
- lookup_table_gcam_nc_files.csv
- lookup_table_globiom_nc_files.csv
- lookup_table_image_nc_files.csv
- lookup_table_image_nc_files_preprocessing.csv
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
Spatially-explicit analysis outputs can be made available upon reasonable request.
Underlying data on climate refugia and land use from GLOBIOM and IMAGE can be made available upon
reasonable request by the individual modelling teams.

Underlying data on land use from AIM-SSP/RCP Ver2018 is available at:
http://doi.org/10.18959/20180403.001

Underlying data on land use from GCAM-Demeter is available at:
https://doi.org/10.25584/data.2020-07.1357/1644253

Underlying data on world administrative boundaries is available at:
https://geonode.wfp.org/layers/geonode%3Awld_bnd_adm0_wfp

# License
**Creative Commons Attribution 4.0 International**.
