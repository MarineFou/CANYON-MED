# CANYON-MED without decimal year

This directory contains a retrained version of CANYON-MED for nutrient estimation, developed to reduce potential biases in predictions beyond 2018, the last year included in the original training dataset.

In this version, decimal year was removed from the input variables to limit potential biases associated with temporal extrapolation.

The model was retrained using the same hyperparameters as the original CANYON-MED implementation. The network weights were updated during retraining, while the ensemble size was maintained at 10 neural networks. However, the individual networks selected for the ensemble may differ from those in the original version.

The original CANYON-MED versions remain unchanged and are available in the main repository.

## Usage

The model files are provided in this directory.

## Citation

If you use this retrained version of CANYON-MED without decimal year in your research, please cite the original CANYON-MED publications:

- Fourrier et al. (2020). *A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED*. Frontiers in Marine Science, 7. https://doi.org/10.3389/fmars.2020.00620

- Fourrier et al. (2021). *Corrigendum: A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED*. Frontiers in Marine Science, 8. https://doi.org/10.3389/fmars.2021.650509

In addition, please acknowledge the use of this retrained version, which excludes decimal year from the input variables and uses updated neural network weights.
