# What is Image-Based Meta- & Mega-Analysis (IBMMA)?

**IBMMA** is a powerful and versatile free software package designed for meta- and mega-analysis on neuroimaging datasets aggregated from multiple study sites, such as ENIGMA (Enhancing NeuroImaging Genetics through Meta Analysis) Consortium and NCANDA (National Consortium on Alcohol and Neurodevelopment in Adolescence). It employs _**mass-univariate**_ statistical models to analyze diverse neuroimaging features, including voxel-, vertex-, and connectome-based anatomical and functional brain measures.

IBMMA harnesses the power of parallel processing by leveraging multi-CPU capabilities available in modern clusters and personal computers. It boasts the remarkable ability to perform statistical analysis on thousands of subjects and millions of neuroimaging features simultaneously, across various platforms such as Linux, Mac, and Windows. This capability is crucial for analyzing big neuroimaging datasets that far exceed the scale of most single cohort studies prevalent in the past two decades.

One of IBMMA's strengths lies in its modeling flexibility. It not only employs widely-used linear models but also incorporates more sophisticated statistical approaches available in released R and Python packages. This versatility enables researchers to tailor their analyses to specific research questions and data characteristics.

IBMMA produces multiple statistical outputs and model estimations that are consistent with the raw data in terms of dimensionality. This consistency ensures that the results are easily visualized and compared across different models, facilitating comprehensive data analysis and interpretation.

In summary, IBMMA is an optimal tool for big neuroimaging data analysis and displaying statistical results. Its powerful capabilities, versatility, and user-friendly output make it an indispensable asset for researchers working with large-scale neuroimaging datasets from multiple study sites.

The current version was mainly developped and tested through using the VSCode(Visual Studio Code) software under Linux platform running on Duke BIAC Cluster. More tests and debugs are needed for different platforms and operating systems. Please download & test the code. Please feel free to contact me at **_ds366@duke.edu_** if you meet any problem.

# Why to Use It?

Neuroimaging studies on large datasets aggregated from multiple sites (or sources) are becoming increasingly popular. Data repositories such as ENIGMA and NCANDA often provide brain data from hundreds or thousands of individuals. Findings from these large neuroimaging datasets are more robust and replicable than results from traditional single-cohort studies, potentially paving the way for uncovering stable biomarkers for clinical diagnosis and intervention. However, advantages also accompany these challenges:

---- **_Meta-Analysis & Mega-Analysis_**. Neuroimaging data were acquired using various scanners and diverse scanning protocols. Demographic characteristics (e.g., age, sex, education level, and socioeconomic status) and clinical characteristics (e.g., diagnosis, comorbidity, and substance/medication usage) of the participants varied significantly across studies. IBMMA employs **meta-** and **mega-analysis** to examine the effects of interest while removing or controlling for the effects of study sites.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_Meta_Mega.png?raw=true)

We can get robust and beautiful Meta- and Mega-analysis results:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_IBMMA_Meta_Mega_Results.png?raw=true)

We can also compare results of different data types or preprocessing protocals:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_Multi_Data.png?raw=true)

---- **_Data Flexibility_**. Neuroimaging data comes in various formats, including 3D/4D images of brain structures and functions, 2D matrices of brain connections between different regions, and 1D data such as values extracted from regions of interest (ROI). Numerous software packages exist, each targeting specific data types. Consequently, researchers often invest significant time and resources learning to use these different tools. IBMMA addresses this challenge by providing a uniform pipeline to analyze all data types consistently. Specifically, IBMMA flattens any type of data into a 1D format and performs statistical analyses on the elements of this 1D data using parallel processing.

---- **_Missing Values_**. Missing values are common in neuroimaging studies due to diverse scanning protocols and low data quality in certain brain regions (such as the ventral prefrontal cortex and ventral temporal lobe). The voxels with missing values differ slightly among individuals from the same study site but may vary significantly among participants from different sites. Widely used software for statistical analysis of neuroimaging data, such as SPM and FSL, cannot handle missing values correctly. In other words, if a single person has a missing value in a particular voxel, there will be no statistical output for this voxel even if thousands of participants are included in the analysis. Some preprocessing software, such as HalfPipe, fills missing values with zeros. This approach slightly biases the results per study site but may significantly influence the outputs of multi-cohort studies, especially in data associated with brain regions that differ greatly in missing values among study sites. IBMMA addresses this issue by running voxel-specific statistical modeling and providing statistical outputs of important characteristics (e.g., number of observations and degrees of freedom) per element (e.g., a voxel or a connection).

Below is an example of the inclusive masks of three participants from the same study site. It is clear that they are significantly different in brain regions especially in the ventral areas of the brain.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_Missing_Values.png?raw=true)

---- **_Model Flexibility_**. The most popular statistical models in neuroimaging studies are based on linear regressions. These typically use linear combinations of diagnosis, age, sex, and other covariates to predict brain signals across voxels. For instance: _Brain ~ Diagnosis + Age + Sex_. However, researchers need the ability to create more flexible statistical models to address diverse research questions. Examples include: 

(a) Using linear or non-linear combinations of brain data, demographic, and clinical covariates to predict diagnosis or symptom severity: _Diagnosis ~ Brain + Age + Sex_; 

(b) Utilizing brain data from multiple modalities: _Symptom Severity ~ MRI + rs-fMRI + task-fMRI + Age + Sex_; 

(c) Considering more than one outcome simultaneously, such as in survival analysis: _Surv(time, status) ~ Brain + covariates_. 

IBMMA can run multiple statistical models by calling released R and Python packages. It simplifies model settings by reading model formulas listed in '**_path_para.xlsx_**'.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_Models.png?raw=true)

# How to Install It?

IBMMA is still in its early stages of development. Users should download the entire folder and extract it to a location on their computer (the working space). Ensure that this folder contains a file called '**_path_para.xlsx_**' (a template for users to modify according to their own data), a file called '**_ibmma.py_**', and a folder called '**_SDL_functions_**'.

----**_Demographic & Clinical File Path_**. "FILE" lists the paths to the .xlsx file of demographic & clinical information, and the corresponding sheet name (listed in "SHEET") in this .xlsx file. The switch "EXCLUDED" (1=excluded, 0=included for analysis) is used to mark the rows that would NOT be included in analysis.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_demographic_clinical.png?raw=true)

----**_Data Path_**. IBMMA does search automatically and recursively across all of the subfolders under the folders with specified "PATH" for the files of interest. The switch "EXCLUDED" (1=excluded, 0=included for analysis) is used to mark the rows that would NOT be included in analysis.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_data_path.png?raw=true)

----**_Data Pattern_**. IBMMA search across the data paths for the files with keywords in their filenames. You MUST define the keywords for "DATA", and may also define the keywords for "INFO" (joson files that usually contain data information such as TR of MRI/fMRI scanning) and "MASK" (i.e., subject-specific mask images). The switch "EXCLUDED" (1=excluded, 0=included for analysis) is used to mark the rows that would NOT be included in analysis.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_data_pattern.png?raw=true)

----**_Predictors_**. It helps to rename and filter the variables / variable levels of interest. "VAR" lists the variable names in the model formula, and "NAME" lists the corresponding variable names in the demographic and clinical document. "OLD2NEW" lists the conversions from the old variable levels in "NAME" and the corresponding new levels in "VAR". "VALUES" lists the variable values to be included in analysis. The cases that the corresponding variable values are not listed here will be removed from analysis. The switch "EXCLUDED" (1=excluded, 0=included for analysis) is used to mark the rows that would NOT be included in analysis.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_predictors.png?raw=true)

----**_Models_**. IBMMA now recognizes model formulas in R style. You do not need to set up the design matrix and contrasts by yourself.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_models.png?raw=true)

# How Does It Work?

IBMMA has a pipeline to automatically run all steps of Meta-analysis (under development) & Mega-analysis. After installation, the users could run IBMMA by entering "**_python ibmma.py_**" in the terminal and wait for the final outcomes. IBMMA runs across data patterns (i.e., rows in the sheet "**_data_pattern_**") and statistical models (i.e., rows in the sheet "**_models_**"). Please be patient to large datasets from multiple study sites, with diverse data types, and multiple statistical models. You may spend several hours to get the final outputs.

**Very important: _fID_**: The subject's ID that is unique in the same dataset may be duplicated in data from different sites. For example, you may find sub_001.nii.gz in both study site A and study site B. Therefore, we need a new type of ID per observation. Here, "**_fID_**" is the combination of site name and subject's ID, e.g., Duke_sub001. IBMMA uses fID to match the demographic/ clinical information and the corresponding neuroimaging data files.

**Step 1**: IBMMA search all files (data, mask, and information) based on their paths and patterns (part of the filenames) listed in "path_para.xlsx". It also generates a new folder "**_Processes_**" to contain all of the temporal outputs. It generates a file called "**_Subjects.csv_**" under the folder "**_Processes_**" to store demographic, clinical, and data (paths to data) information.

-- Sheet "**_demographic_clinical_**" indicates the path to the .xlsx file and the name of the sheet for demographic and clinical information. 

-- Sheet "**_data_path_**" lists the path(s) to the folders of data that were the outputs of some preprocessing softwares such as **_HALFPIPE_**. 

-- Sheet "**_data_pattern_**" lists the pattern (part of a file name) of files of interest. For example, "_feature-fALFF_alff.nii.gz" for NIFTI data, "_feature-fALFF_alff.json" contains the important information to the corresponding NIFTI data, and "_feature-fALFF_mask.nii.gz" refers to the corresponding NIFTI mask.

-- Sheet "**_predictors_**" lists all variables that appear in the statistical model(s). The users can match the variables in the models to the variables listed in the file of demograohic and clinical information, and they can also match the values per variable.

-- Sheet "**_models_**" are the statistical models used for the analyses. The model formula follows R algorithm. The users do not need to figure out the complex design matrix and contrasts by themselves.


**Step 2**: IBMMA generates a new folder "**_masked_**" under the folder "**_Processes_**" and masks the data files (whatever NIFTI images or adjacent matrix saved in .csv files) using the mask files that have the same dimension as the data files. This step is important because some preprocessing softwares impute missing values (due to no information or low-quality values in the corresponding voxel or connection) with 0s. That may lead to wrong statistical outputs, especially for meta- & mega-analysis that are targetting data from different study sites. If there is no mask file, the data file will be used instead.

**Step 3**: IBMMA generates a new folder "**_flattened_**" under the folder "**_Processes_**" and flattens any kind of data into one-dimension.

**Step 4**: IBMMA generates a new folder "**_segmented_**" under the folder "**_Processes_**" and  extracts the _i_ th segment of the flattened data across subjects and vertically combine them into a new CSV file in which each row represents a subject. The default number of segmentation is _50_.

**Step 5**: IBMMA generates a new folder "**_stats_**" under the folder "**_Processes_**" and  runs statistical modelling by calling R or Python scripts to run parallel analysis. There are two types of statistical outputs: TIDY, which includs the information that are often listed in reports and articles (such as regression coefficients, degree of freedom, T values, p values); and GLANCE, which includes estimates of model fitting (such as AIC, BIC, and number of observations). TIDY & GLANCE are from R package _broom_.
It should be noted that, in some high performance computer (HPC) or cluster, The users need to load some module before running IBMMA. For example, enter "**_module load R/latest_**" if R was not explicited installed in your path.

**Step 6**: IBMMA generates a new folder folder "**_Results_**" and concatenates the statistical outputs from different degments into one.

**Step 7**: IBMMA reverses the concatenated statistical outputs back to the original dimensions of the input data. That is to say, statistical analyses outputs for NIFTI image are still NIFTI images, and for adjacent matrix are still matrix. This step also includes FDR correction for multiple comparisons as well as negatively log10 transformed p-values for observation purpose.

# What Are The Outputs

In the folder "**_Results_**", users could find the subfolders with name listed in the sheet "**_data_pattern_**" column "**_NAME_**". Within each subfolder, there are two folders named "**_Mega_**" and "**_Meta_**" (under development). Within "**_Mega_**", there are two subfolder "**_TIDY_**" and "**_GLANCE_**".

In the folder "**_TIDY_**", there are subfolders: 

---- "**_df_**": Degree of Freedom.

---- "**_estimate_**": Regression coefficient.

---- "**_p.value_**": Uncorrected p value.

---- "**_p.value.fdr_**": FDR corrected p value.

---- "**_statistic_**": t value.

---- "**_std.error_**": standrad error. 

For example, TIDY outputs for brain images:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_TIDY.png?raw=true)

For another example, TIDY outputs for correlation matrix (symmetric) of ROI-to-ROI functional connectivity:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_TIDY_Matrix.png?raw=true)

It also provides FDR_corrected p values and log10() transformed p values for observation purposes:

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_TIDY_Matrix_ps.png?raw=true)

In the folder "**_GLANCE_**", there are subfolders: 

---- "**_AIC_**": Akaike Information Criterion. A measure of model quality that balances goodness of fit with model simplicity. Lower AIC values indicate better models.

---- "**_BIC_**": Bayesian Information Criterion. Similar to AIC, but penalizes model complexity more strongly. Also used for model selection, with lower values being better.

---- "**_df.residual_**": Degrees of Freedom Residual. The number of observations minus the number of parameters estimated in the model. It represents the remaining degrees of freedom after fitting the model.

---- "**_logLik_**": Log-Likelihood. The logarithm of the likelihood function, which measures how well the model fits the observed data.

---- "**_nobs_**": Number of observations. The total number of data points used in the analysis. It is very important to understand how the findings are influenced by the sample size in big data analysis.

---- "**_REMLcrit_**": Restricted Maximum Likelihood Criterion. A criterion used in mixed-effects models for estimating variance components. It's an alternative to maximum likelihood estimation.

---- "**_sigma_**": Residual Standard Error. An estimate of the standard deviation of the residuals in a regression model. It measures the average distance between the observed values and the predicted values.

For example, GLANCE outputs for brain images:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_GLANCE.png?raw=true)

For another example, GLANCE outputs for correlation matrix (symmetric) of ROI-to-ROI functional connectivity:
![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_GLANCE_Matrix.png?raw=true)

# What Will Come in Next Versions?

Several modules and functions have been planned to be incorporated into IBMMA in future versions:

---- Meta analysis section. There is an earlier version of the Meta-analysis in IBMMA (https://github.com/sundelinustc/ENIGMA_PTSD_MDD). The new version of meta-analysis will apply the algorithm similar to the released Mega-analysis.

![alt text](https://github.com/sundelinustc/IBMMA/blob/main/Figures/Fig_IBMMA_Meta_Results.png?raw=true)

---- Multiple methods of correction for multiple comparisons. Now, the default method is FDR whatever the data type is. pTFCE for voxel-wised NIFTI images (https://spisakt.github.io/pTFCE/) has been incorporated. NBS for connection matrix will be added.

---- Multiple data format. The current version is for the functional neuroimaging data (voxel-based images in *.nii.gz format or connectom-based matrix in .csv format) preprocessed by softwares such as HALFPIPE. The updated IBMMA version will have the ability to do analyses on any format of neuroimaging data, such as cortical thickness, surface area, and gyrification.

---- Effect size. Calculations of efect size are important in many aspects such as Power analysis. Future version of IBMMA may include the function to calculate the whole-brain map and matrix of effect size. However, calculation of effect size may be tricky in some models such as linear mixed effects models.

---- Measures module. The current IBMMA is designed for statistical analysis on data preprocessed by softwares such as HALFPIPE. A new module may be included to get measures (e.g., static and dynamic functional connectivity, community entropy) based on the preprocessed data and enter them to statistical modellings.

---- Normative modeling. Normative modeling is an emerging and innovative framework for mapping individual differences at the level of a single subject or observation in relation to a reference model (see https://www.nature.com/articles/s41596-022-00696-5). Future versions of IBMMA may add this function to chart centiles of variation across a population in terms of mappings between biology and behavior, which can then be used to make statistical inferences at the level of the individual. 

---- Collaboration with **_DS (Deep Simple)_** package (https://github.com/sundelinustc/Deep_Simple) to promote manuscript preparations.

---- Distributed computing resources. Take advantage of parallel processing across multiple nodes or cores using R packages such as Rmpi (https://cran.r-project.org/web/packages/Rmpi/index.html).




