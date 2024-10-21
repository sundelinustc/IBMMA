# This R script runs pTFCE (Probabilistic Threshold-free Cluster Enhancement)
# https://spisakt.github.io/pTFCE/
# https://github.com/spisakt/pTFCE/wiki/3.-R-package
# Tamás Spisák, Zsófia Spisák, Matthias Zunhammer, Ulrike Bingel, Stephen Smith, Thomas Nichols, Tamás Kincses, Probabilistic TFCE: a generalised combination of cluster size and voxel intensity to increase statistical power. Neuroimage, 185:12-26. DOI: 10.1016/j.neuroimage.2018.09.078

# This script could be run using terminal command with 5 arguments such as
# Rscript path/to/R_modelling_parallel.R arg1 arg2 arg3 arg4 arg5
# --- arg1, full path to xfile.csv, which contains predictors (row = observation, column = variable).
# --- arg2, full path to yfile.csv, which contains outcome variables (row = observation, column = variable).
# --- arg3, full path to the output directory.
# --- arg4, model name, e.g. "model_01"
# --- arg5, texts of model formula, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))" or "lm(Yvar ~ GROUP + AGE + SEX)".
# An example of terminal command:
# Rscript ./SDL_functions/R_modelling_parallel.R ./Progress/Subjects.csv ./Progress/atlas_conn_SFC/static_fc/segmented/V47000_48880.csv  ./Progress/atlas_conn_SFC/static_fc/stats/V47000_48880/Mega "model_01" "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))"

# Copyright (c) 2024 Delin Sun (ds366@duke.edu; sundelinustc@gmail.com)
# All rights reserved.
# This script is the property of Delin Sun.
# No part of this script may be reproduced in any form without the prior permission of Delin Sun.

# for test purpose only
path_Tmap  <- '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA_v0.1.1-beta/Results/reHo/Mega/TIDY/statistic/GROUP/OUT_Model_01.nii.gz'
path_DoF   <- '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA_v0.1.1-beta/Results/reHo/Mega/TIDY/df/GROUP/OUT_Model_01.nii.gz'
path_Mask  <- '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA_v0.1.1-beta/SDL_functions/brain_mask.nii'

# (0) get command line arguments
args <- commandArgs(trailingOnly = TRUE)
path_Tmap   <- args[1]
path_DoF   <- args[2]
path_Mask <- args[3]

# (1) Packages
# install pacman to load the other packages
if (!require(pacman)){install.packages('pacman', dependencies = TRUE)}
library(pacman)
# basic packages
packages <- c('pTFCE','oro.nifti','RNifti')
# load all packages of interest
do.call(p_load, as.list(packages))

# # (optional) if pTFCE has not been installed
# packages <- c('devtools','oro.nifti','RNifti')
# # load all packages of interest
# do.call(p_load, as.list(packages))
# # install pTFCE
# if (!requireNamespace("pTFCE", quietly = TRUE)) {
#   install.packages("remotes")
#   remotes::install_github("spisakt/pTFCE@v0.2.2.1")
# }
# library(pTFCE)

# (2) Helper functions
# T map
Tmap <- readNIfTI(path_Tmap)
# Output folder
Out_folder <- paste0(path_Tmap,'_pTFCE')
if (!dir.exists(Out_folder)) {dir.create(Out_folder, recursive = TRUE)}
# Degrees of freedom
DoF <- readNIfTI(path_DoF)
# Mask
Mask <- readNIfTI(path_Mask)

# T-to-Z conversion
Z <- qnorm(pt(Tmap, df=DoF, log.p = TRUE), log.p = TRUE) # did it actually use the df map or just a single df value?
writeNIfTI(Z, file.path(Out_folder, 'Zmap')) # Save the Zmap

# Reversed Z
Z_reversed <- Z
Z_reversed@.Data <- -1 * Z@.Data
Z_reversed@descrip <- paste0(Z@descrip, " (Reversed)")
writeNIfTI(Z_reversed, file.path(Out_folder, 'Zmap_reversed')) # Save the Zmap # Write the reversed Z-map

# pTFCE of Z
pTFCE <- ptfce(Z, Mask)
writeNIfTI(pTFCE$Z, file.path(Out_folder, 'pTFCE-z-score-map')) # save
line <- sprintf("%.3f: Z threshold for pTFCE-z-score-map after FWER correction (p < 0.05)", pTFCE$fwer0.05.Z)
writeLines(line, file.path(Out_folder, 'thres_z_fwer0.05.txt'))

# pTFCE of Z_reversed
pTFCE <- ptfce(Z_reversed, Mask)
writeNIfTI(pTFCE$Z, file.path(Out_folder, 'pTFCE-z-reversed-score-map')) # save
line <- sprintf("%.3f: Z threshold for pTFCE-z-reversed-score-map after FWER correction (p < 0.05)", pTFCE$fwer0.05.Z)
write(line, file = file.path(Out_folder, 'thres_z_fwer0.05.txt'), append = TRUE)
