The sample data include simulated voxel-wise brain activity (see nifti subfolder) and atlas-based brain connectivity (see matrix subfolder) from 24 subjects across 3 study sites (see Demographic.xlsx). 

File paths and analysis parameters are specified in path_para.xlsx (note: you will need to update the paths to match your local directory structure).

The sample statistical model includes fixed effects of group (case vs. control, coded as 1 vs. 0), sex (female vs. male, coded as 1 vs. 0), age (linear term), and age² (quadratic term to model nonlinear age effects), as well as a random intercept for study site.

Analysis of the simulated data reveals significant group effects in:

Brain activity: Posterior cingulate cortex (FWE p < 0.05 within whole-brain grey matter mask tpl-MNI152NLin2009cAsym_res-02_label-GM_binary_mask_80percent; see Reports/nifti_sample/Mega/group/Model_01/index.html)

Connectivity: Two prefrontal connectivity measures (FDR p < 0.05 using the Brainnetome atlas; atlas regions and labels are in tpl-MNI152NLin2009cAsym_atlas-brainnetome_dseg.nii.gz and tpl-MNI152NLin2009cAsym_atlas-brainnetome_dseg.txt, respectively; see Reports/matrix_sample/Mega/group/Model_01/index.html)