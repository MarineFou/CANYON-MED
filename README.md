# CANYON-MED
**Matlab and R** implementations of the CANYON-MED method published in:

Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2020). A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 7. doi:10.3389/fmars.2020.00620.

Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2021). Corrigendum: A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 8. doi:10.3389/fmars.2021.650509.

**Python** code has developped after publication. Please refer to the papers mentionned above and the github repository if using the python code.
When using the method, please **cite the papers**.
For any questions, please contact me at : *marine.fourrier@protonmail.com*

------
UPDATE: THE PYTHON CODE IS NOW FUNCTIONAL!
------

Matlab, R and Python routines to estimate open ocean carbonate system variables and nutrients in the Mediterranean Sea from time, geolocation, together with T, S and O2.

The folders for Matlab, R and Python each contain the CANYON-MED codes and training weights.

To use the CANYON-MED neural networks, download the corresponding folder ('CANYON-MED/v2/').
In the "CANYON-MED codes" folder, change the **"basedir" in all 6 CANYON-MED functions** to the appropriate folder on your computer. This folder is the location of the CANYON-MED folder.
It has to end in "R/", "MATLAB/" or "PYTHON/".

For Matlab users, you will need to add the entire folder containing the code to your path in order for the custom functions to be recognised. One way to do this is to go to the folder containing the CANYON-MED codes and type **(addpath(genpath(pwd))** in your command window.

To verify the proper installation, apply the function with the **check values** (i.e. 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1) and check that the value corresponds to the value line 22.

**Do not change the architecture inside the folder or rename any of the files.**

------

Modifications mentionned in the corrigendum :
The inputs have been modified (the day of year and year are now considered as a unique input : the decimal year). A retraining has been done on 2 separate subsets to diversify the learning process.
The sigmoid transformation of the day of year is no longer used, thus removing a problem with seasonnality at depth.

**For users who don't have the Aerospace Toolbox, the decimal year and leap year functions are not available. These have been replaced by a few lines of code that provide the same functionality. Simply comment line 35 and uncomment lines 39-51 in all 6 CANYON-MED functions.**
------

The folder v1_paper contains the original version from the paper.
**As some errors have been modified we recommend users do not use this version.**

