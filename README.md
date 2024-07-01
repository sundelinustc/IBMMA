# Background

**Image-Based Meta- & Mega-Analysis (IBMMA)** is a powerful and versatile free software package designed for meta- and mega-analysis on neuroimaging datasets aggregated from multiple study sites, such as ENIGMA (Enhancing NeuroImaging Genetics through Meta Analysis) Consortium and NCANDA (National Consortium on Alcohol and Neurodevelopment in Adolescence). It employs _**mass-univariate**_ statistical models to analyze diverse neuroimaging features, including voxel-, vertex-, and connectome-based anatomical and functional brain measures.

IBMMA harnesses the power of parallel processing by leveraging multi-CPU capabilities available in modern clusters and personal computers. It boasts the remarkable ability to perform statistical analysis on thousands of subjects and millions of neuroimaging features simultaneously, across various platforms such as Linux, Mac, and Windows. This capability is crucial for analyzing big neuroimaging datasets that far exceed the scale of most single cohort studies prevalent in the past two decades.

One of IBMMA's strengths lies in its modeling flexibility. It not only employs widely-used linear models but also incorporates more sophisticated statistical approaches available in released R and Python packages. This versatility enables researchers to tailor their analyses to specific research questions and data characteristics.

IBMMA produces multiple statistical outputs and model estimations that are consistent with the raw data in terms of dimensionality. This consistency ensures that the results are easily visualized and compared across different models, facilitating comprehensive data analysis and interpretation.

In summary, IBMMA is an optimal tool for big neuroimaging data analysis and displaying statistical results. Its powerful capabilities, versatility, and user-friendly output make it an indispensable asset for researchers working with large-scale neuroimaging datasets from multiple study sites.

# How Does it Work

IBMMA has a pipeline to automatically run all steps of Meta- & Mega-analysis. The users just need to dowload the package and setup the "path_para.xlsx" file correctly.

**Step 1**: IBMMA search all files (data, mask, and information) based on their paths and patterns (part of the filenames) listed in "path_para.xlsx". 

-- Sheet "demographic_clinical" indicates the path to the .xlsx file and the name of the sheet for demographic and clinical information. 

-- Sheet "data_path" lists the path(s) to the folders of data that were the outputs of some preprocessing softwares such as HalfPipe. 

-- Sheet "data_pattern" lists the pattern (part of a file name) of files of interest. For example, "_feature-fALFF_alff.nii.gz" for NIFTI data, "_feature-fALFF_alff.json" contains the important information to the corresponding NIFTI data, and "_feature-fALFF_mask.nii.gz" refers to the corresponding NIFTI mask.

-- Sheet "predictors" lists all variables that appear in the statistical model(s). The users can match the variables in the models to the variables listed in the file of demograohic and clinical information, and they can also match the values per variable.

-- Sheet "models" are the statistical models used for the analyses. The model formula follows R algorithm. The users do not need to figure out the conplex design matrix and contrasts by themselves.


**Step 2**: 

# Installation

IBMMA is still at its beta version. Users please download the whole folder and unpack it somewhere (the working space) in your computer. Make sure that, in this folder, there are a file called "path_para.xlsx" and a folder called "SDL_functions".

# File Structure

## path_para.xlsx
This file contains all information of paths and parameters for statistical analysis. It has a few sheets:
### **demographic_clinical**

