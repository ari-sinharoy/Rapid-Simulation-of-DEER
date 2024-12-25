# Rapid Simulation of DEER
### DEER kernel derivation, expression and experimental data 

### The **Code** folder contains:
- Expression of the derived full kernel (*deer_newkernel.py*)
- P(r) determination using the full kernel + Tikhonov regularization + L-curve (*pr_derivation.py*)
- P(r) determination using the full kernel + Tikhonov regularization + L-curve with Gaussian P(theta) for biradical-I (*pr_derivation_birad-I.py*)
- Time-domain simulations for biradical-I using the full kernel (*deer_simu_birad-I.py*)
- Calculation of P(theta) for biradical-I and their Gaussian fits (*ptheta_calc_birad-I.py*, *ptheta_fits_birad-I.py*) 
- Methods to find an optimal parameter for Tikhonov regularization (*TIKR_methods.py*)
- DEERLab script for background fitting and P(r) determination (*deerlab_pr_derivation.py*)

### The **Data** folder contains:
- Experimental DEER data for the three biradicals (subfolders: *DEER Data_biradical-I*, *DEER Data_biradical-II*, *DEER Data_biradical-III*)
- P(r) obtained using the full kernel (*NewKernel_Pr*)
- Time-domain simulations for biradical-I using the full kernel (*NewKernel_Simulations*)
- Calculated P(theta)'s for biradical-I (*Ptheta_Calculation*)
- Background fitted DEER data and P(r)'s by DEERLab (*DEERLab_Analysis*) 

#### Cite **https://doi.org/10.1021/acs.jpclett.4c03245** to refer to this work
