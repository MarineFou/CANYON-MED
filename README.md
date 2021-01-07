# CANYON-MED
Matlab and R implementations of the CANYON-MED method published in:

Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2020). A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 7. doi:10.3389/fmars.2020.00620.


When using the method, please cite the paper.
For any questions, please contact me at : marine.fourrier@protonmail.com

------

Matlab and R routines to estimate open ocean carbonate system variables and nutrients in the Mediterranean Sea from time, geolocation, together with T, S and O2.

The folders for Matlab and R each contain the CANYON-MED codes and training weights.

To use the CANYON-MED neural networks, download the corresponding folder ('CANYON-MED/v1_paper/').
In the "CANYON-MED codes" folder, change the "basedir" in all 6 CANYON-MED functions to the appropriate folder on your computer. This folder is the location of the CANYON-MED folder.
It has to end in "R" or "MATLAB".

To verify the proper installation, apply the function with the check values (i.e. 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1) and check that the value corresponds to the value line 22.

Do not change the architecture inside the folder or rename any of the files.

------

The folder v2 contains a second version. The installation process is the same as for v1_paper.


The inputs have been modified (the day of year and year are now considered as a unique input : the decimal year). A retraining has been done on 2 separate subsets to diversify the learning process.
The sigmoid transformation of the day of year is no longer used, thus removing a problem with seasonnality at depth.

A correction of the paper is under way. WeWE RECOMMEND USERS USE v2.
