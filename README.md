# TAGI-V
This repository presents the MATLAB package for TAGI-V for handling heteroscedastic aleatory uncertainty for regression tasks. The package is built on top of the original package for TAGI which can be found here [https://github.com/CivML-PolyMtl/TAGI].  
The package displays its capability of modeling heteroscedasticity through a toy problem in the Toy Problems folder (inside the regression folder). Moreover, the package performs tests on real UCI regression datasets to show its predictive capacity. The package uses small UCI datasets such as Boston, Yacht, as well as large datasets such Elevetor, KeggDirected to show its performance compared to other existing approximate inference methods. The codes for running TAGI-V for each dataset can be found inside the folder corresponding to that dataset. For e.g. to run TAGI-V for BostonHousing, go to the folder and run the MATLAB file BostonHousing.m. Similarly, for Concrete and Energy it is concrete.m and energy.m

# References
*Tractable Approximate Gaussian Inference for Bayesian Neural Networks*
Goulet, J.-A., Nguyen, L.H., and Amiri, S.
Journal of Machine Learning Research, 2021, 20-1009, Volume 22, Number 251, pp. 1-23.
