import os  # for file operations
import numpy as np # for manipulating arrays
import pandas as pd # for working with csv files
import nibabel as nib # for reading and writing nifti images
from nilearn import image # for image resampling
import nilearn.plotting as plotting # for plotting matrix of connection
import matplotlib.pyplot as plt # for making the frame of plotting
import multiprocessing as mp # for parallel processing
import subprocess # for calling functions through OS system
import time # for measuring time elapsed
from tqdm import tqdm # for progress bar
# from statsmodels.stats.multitest import multipletests # for correction of multiple comparisons
# from scipy import stats 

# Function to extract mean time series from a 4D NIfTI image based on ROIs defined in a list of mask images.
def ROI_extract_mean_timeseries(func_img_path, mask_img_paths, roi_labels):
    """    
    Args:
    func_img_path (str):   Path to the 4D functional NIfTI image.
    mask_img_paths (list): List of paths to 3D mask NIfTI images with ROIs.
    roi_labels (list):     List of labels of ROIs (usually from the 1st subject).
    
    Returns:
    pd.DataFrame:          DataFrame with mean time series for each ROI
    """
    
    # Load the functional image
    func_img = nib.load(func_img_path)
    func_data = func_img.get_fdata()
    
    # Initialize a dictionary to store time series
    timeseries_dict = {}
    
    # Loop over each mask image path
    for ROI_index, mask_img_path in enumerate(mask_img_paths, start=1):
        # Load the mask image
        mask_img = nib.load(mask_img_path)
        mask_data = mask_img.get_fdata()
        
        # Check if the spatial dimensions match
        if func_data.shape[:3] != mask_data.shape:
            print("Resample mask image to the dimensions of functional image given that their dimensions do not match!!!")
            
            # Resample mask to match functional image dimensions
            resampled_mask_img = image.resample_to_img(mask_img, func_img, interpolation='nearest')
        
            # Get the new mask data as numpy arrays
            mask_data = resampled_mask_img.get_fdata()
        
        # Find unique ROI values in the mask
        roi_values = np.unique(mask_data)
        roi_values = roi_values[roi_values != 0]  # Exclude 0 (typically background)
        
        # Extract mean time series for each ROI
        for roi in roi_values:
            roi_mask = mask_data == roi
            roi_timeseries = np.nanmean(func_data[roi_mask], axis=0)  # Use np.nanmean to ignore nan values
            
            # Create a column name with filename and ROI name
            # filename = os.path.basename(mask_img_path)
            column_name = f'ROI{ROI_index}_roi_{int(roi)}' # e.g., tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.nii.gz_ROI_3
            
            timeseries_dict[column_name] = roi_timeseries
    
    # Convert to DataFrame
    df = pd.DataFrame(timeseries_dict)
    
    # Match pre-existed ROI labels (usually from the 1st subject)
    if roi_labels:
        # Create a new DataFrame (full of NaN) with the same index as df and columns from roi_labels
        new_df = pd.DataFrame(index=df.index, columns=roi_labels)
        # Update new_df with data from df where columns match
        new_df.update(df[df.columns.intersection(roi_labels)])
        # Reorder columns to match roi_labels
        new_df = new_df[roi_labels]
        # Replace df with the new DataFrame
        df = new_df
    
    return df

# Function to parse numbers from a text string such as "5:8,10:15"
def parse_number_array(text):
    parts = text.split(',')
    numbers = []
    for part in parts:
        if ':' in part:
            # Handle ranges like "1:5" or "16:17"
            start, end = map(int, part.split(':'))
            numbers.extend(range(start, end + 1))
        else:
            # Handle single numbers
            numbers.append(int(part))
    return numbers
# # Function to extract data from ROIs
# def ROI_data_extraction(args):
#     """
#     Args:
#         args (tuple): A tuple containing the data file path, ROI file path.
    
#     Returns:
#         df: A dataframe of extracted data. Each column represents a ROI. 
#             There is only one row of values if the dimensions are the same between the data and the ROIs, e.g. 3D image of ROIs applied to 3D image of data.
#             While there are many rows of values if the dimensions of data are greater than that of the ROIs, e.g. 3D image of ROIs applied to 4D image of time series. In this example, each row corresponds to a time point.
#     """
    
# Function for calculating static functional connectivity (SFC)
def SFC(df):
    """
    Args:
        df (dataframe): A dataframe of time series extracted from ROIs. Each column represents a ROI.
    
    Returns:
        z_matrix (dataframe): A dataframe of R-to-Z transformed (Pearson's) correlation matrix.
    """
    
    # Calculate pairwise correlations
    corr_matrix = df.corr(method='pearson')
    
    # Apply Fisher's r-to-z transformation
    # It converts the sampling distribution of Pearson's r from a skewed distribution to an approximately normal distribution.
    # This normalization allows for more accurate hypothesis testing and the construction of confidence intervals for correlations.
    # The below "with ..." code is to suppress the warning of "divide by zero encountered in arctanh"
    with np.errstate(divide='ignore', invalid='ignore'):
        df_z_matrix = np.arctanh(corr_matrix) # in fact a dataframe
    
    # Replace infinite values wirth nan
    df_z_matrix = df_z_matrix.replace([np.inf, -np.inf], np.nan)
    
    return df_z_matrix

# Function for calculating unctional connectivity (FC) per subject
def FC_single(args):
    """
    Args:
        func_img_path (str):            Path to the 4D functional NIfTI image
        mask_img_paths (list):          List of paths to 3D mask NIfTI images with ROIs
        process_dir (str):              Path to the directory where the MEGA/META analysis should take.
        fID (str):                      The unique subject's ID: SITE + SubjID.
        roi_labels (list):              List of labels of ROIs. Usually the 1st subject's data.
        roi4roi:                        Path to the file of ROIs' ROI.
        TR (in secs):                   Repetition time (length of a time point, for dynamic functional connectivity only).
        window_length (in seconds):     Length of the sliding window (for dynamic functional connectivity only).
        window_overlap (in percentage): Overlap between adjacent windows (0-1). 1 (default) means SFC.
    Returns:
        z_matrix (dataframe): A dataframe of matrix reflecting SFC (if window_overlap==1) or the SD across multiple SFC (each per time window).
    """
    # Arguments
    func_img_path, mask_img_paths, process_dir, fID, roi_labels, roi4roi, TR, window_length, window_overlap = args
    
    # Extract the time series based on data image and mask images
    df = ROI_extract_mean_timeseries(func_img_path, mask_img_paths, roi_labels)
    
    if window_overlap == 1:
        # Static functional connectivity (SFC)
        fc_df = SFC(df) # call the function of static functional connectivity
        
    else:
        # Dynamic functional connectivity (DFC)
        # Calculate the number of time points in each window
        window_size = int(window_length / TR)
        # Calculate the step size based on the overlap percentage
        step_size = int(window_size * (1 - window_overlap))
        
        # Initialize a list to store the functional connectivity matrices
        fc_matrices = []
        
        # Loop through the time series with the specified window size and step size
        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size
            window_df = df.iloc[start:end]
            fc_matrix = SFC(window_df) # call the function of static functional connectivity
            fc_matrices.append(fc_matrix)
        
        # Convert the list of matrices to a 3D numpy array
        fc_matrices = np.array(fc_matrices)
        
        # Calculate the standard deviation of functional connectivity across time windows
        std_fc = np.std(fc_matrices, axis=0)
        
        # Convert the result to a DataFrame
        fc_df = pd.DataFrame(std_fc, index=df.columns, columns=df.columns)
    
    # ROI4ROI if possible
    if roi4roi:
        my_rois = pd.read_excel(roi4roi, sheet_name='MY_ROIs', dtype='object')
        # Parse number from texts
        my_rois['Row']    = my_rois['Row'].apply(parse_number_array)
        my_rois['Column'] = my_rois['Column'].apply(parse_number_array)
        # Create a dictionary with column names as keys and NaN as values
        data = {col: np.nan for col in my_rois['Label']}
        # Create a DataFrame with one row of NaN values
        df1 = pd.DataFrame(data, index=[0])
        # Calculate per self-defined ROI
        for i in range(len(my_rois)):
            # Select the specified rows and columns
            subset = fc_df.iloc[my_rois['Row'][i], my_rois['Column'][i]]
            # Calculate the mean of self-defined ROI, ignoring NaN values unless all of them are nan
            df1.loc[0, my_rois['Label'][i]] = np.nanmean(subset.values) if subset.notna().any().any() else np.nan
        # replace fc_df
        fc_df = df1
    
    
    # Save
    out_file = os.path.join(process_dir,fID+'.csv')
    fc_df.to_csv(out_file, index=False, header=False, na_rep='NaN') # output NaN to avoid errors when flattening data
    print(f"ROI-to-ROI functional connectivity saved in {out_file}.")
    
    return fc_df

# Class to calculate functional connectivity (SFC & DFC)
class Measures:
    def __init__(self, num_processes=None):
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count() # use all CPUs if not specified

    def Functional_Connectivity(self, subjects_csv_path, data_pattern):  
        """
        Functional connectivity analysis for both static (SFC) & dynamic (DFC) ROI-to-ROI functional connectivity.
        
        Args:
            subjects_csv_path (str): Path to the CSV file containing the data file paths and fIDs.
            data_pattern (series):   A series reflecting a row in the sheet "data_pattern" in "path_para.xlsx"
        
        Outputs:
            save SFC or DFC data per subject into the folder of masked.
        """
        # print information
        t0 = time.time() # start time
        print(f"\nCalculation of functional connectivity (time consuming for multiple files) ... ")
        
        process_dir  = os.path.join('Processes', data_pattern['NAME'])
        # result_dir   = os.path.join('Results', data_pattern['NAME'])
        
        # output_dir
        output_dir = os.path.join(process_dir,'masked')
        # make the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read the data file paths and filename IDs from the CSV file
        subjects_df = pd.read_csv(subjects_csv_path)
        # remove rows that are empty in the columns of fID & col_name
        col_name = 'FULL_PATH_DATA_'+data_pattern['NAME']
        subjects_df.dropna(subset=['fID', col_name], inplace=True)
        # Filter column names that start with "FULL_PATH_ROI" and end with "atlas_SFC"
        filtered_cols = [col for col in subjects_df.columns if col.startswith("FULL_PATH_ROI") and col.endswith(data_pattern['NAME'])]
        # Sort the filtered column names alphabetically
        sorted_cols = sorted(filtered_cols)
        # Drop rows with any empty values in the specified columns (to make sure that all subjects have data from all of the ROIs)
        cleaned_subjects_df = subjects_df.dropna(subset=sorted_cols)
        
        # Arguments
        roi_labels = None # labels of rois, will use the 1st subject's roi labels if not provided
        roi4roi = data_pattern['MYROI'] if data_pattern['MYROI'] and isinstance(data_pattern['MYROI'], (str, bytes, os.PathLike)) and os.path.isfile(data_pattern['MYROI']) else None
        TR=3 # default
        window_length=0 # default
        window_overlap=1 # default
        my_args = [
            (row[col_name],
            pd.Series({idx: row[idx] for idx in sorted_cols if idx in row.index}).tolist(),
            output_dir,
            row['fID'],
            roi_labels,
            roi4roi,
            TR, 
            window_length, 
            window_overlap)
            for _, row in cleaned_subjects_df.iterrows()
        ]
        
        # Get the list of ROI labels
        df = ROI_extract_mean_timeseries(my_args[0][0], my_args[0][1], my_args[0][4])
        roi_labels = df.columns.to_list()
        
        # Re-arguments with updated roi_labels
        my_args = [
            (row[col_name],
            pd.Series({idx: row[idx] for idx in sorted_cols if idx in row.index}).tolist(),
            output_dir,
            row['fID'],
            roi_labels,
            roi4roi,
            TR, 
            window_length, 
            window_overlap)
            for _, row in cleaned_subjects_df.iterrows()
        ]
        
        # # For test purpose
        # FC_single(my_args[10])
        
        # Parallel processing
        # using fewer CPUs to avoid memory problems
        with mp.Pool(processes=self.num_processes//3) as pool:
            pool.map(FC_single, my_args)
            
        # Update Subjects.csv by adding a column of DATA1 to replace the path to masked data files
        subjects_df = pd.read_csv(subjects_csv_path) # Read the Subjects.csv again
        col_name_new = col_name.replace('FULL_PATH_DATA_', 'FULL_PATH_DATA1_') # new column name
        list_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir)] # list of masked files
        base_name, file_extension = os.path.splitext(list_files[0]) # get file extension
        if file_extension.lower() == '.gz' and base_name.lower().endswith('.nii'):
            file_extension = ".nii.gz"
        subjects_df[col_name_new] = subjects_df['fID'].apply(lambda x: os.path.join(output_dir, x+file_extension)) # generate the new column of DATA1 with path/fID.extension 
        subjects_df[col_name_new] = subjects_df[col_name_new].apply(lambda x: '' if x not in list_files else x) # replace values in column 'A' with an empty string if not in the list
        subjects_df.to_csv(subjects_csv_path, index=True) # save into Subjects.csv file in the Processes folder
        print("Subjects.csv updated for the paths to masked data files!")
        
        # Print ending info
        print(f"Calculation of functional connectivity was completed!\nTime elapsed (in secs): {time.time()-t0}\n")

    def pipeline(self, Subjects, data_pattern):
        """
        Pipeline to run measures written in data_pattern['MEASURE']
        
        Args:
            Subjects:     
            data_pattern: A series (row) of the sheet "data_pattern" in "path_para.xlsx"    
        """   
        
        process_dir  = os.path.join('Processes', data_pattern['NAME'])
        result_dir   = os.path.join('Results', data_pattern['NAME'])
              
        if data_pattern['MEASURE'] == 'Functional_Connectivity':
            self.Functional_Connectivity(Subjects, data_pattern)
            