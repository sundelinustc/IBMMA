import os  # for file operations
import nibabel as nib # for reading and writing nifti images
import numpy as np # for manipulating arrays
import pandas as pd # for working with csv files
import nilearn.plotting as plotting # for plotting matrix of connection
import matplotlib.pyplot as plt # for making the frame of plotting
import multiprocessing as mp # for parallel processing
import subprocess # for calling functions through OS system
import time # for measuring time elapsed
from tqdm import tqdm # for progress bar
from statsmodels.stats.multitest import multipletests # for correction of multiple comparisons

# Function to mask data file
def mask_single(args):
    """
    Mask data files.

    Args:
        args (tuple): A tuple containing the data file path, mask file path, the output directory, and the filename ID.
    """
    # Add type checking and debugging information
    if not isinstance(args, tuple) or len(args) != 4:
        print(f"Error: args should be a tuple of 4 elements, received: {args}")
        return

    data_file_path, mask_file_path, output_dir, filename_id = args

    # Print debugging information
    print(f"Debug: data_file_path = {data_file_path}, type = {type(data_file_path)}")
    print(f"Debug: mask_file_path = {mask_file_path}, type = {type(mask_file_path)}")
    print(f"Debug: output_dir = {output_dir}, type = {type(output_dir)}")
    print(f"Debug: filename_id = {filename_id}, type = {type(filename_id)}")

    # Check if the data file exists
    if not isinstance(data_file_path, str) or not os.path.isfile(data_file_path):
        print(f"Error: Invalid data file path '{data_file_path}'.")
        return

    # Get the data file basename & extension
    base_name, file_extension = os.path.splitext(os.path.basename(data_file_path)) 
    # Revise file extension if it is ".nii.gz"
    if file_extension.lower() == '.gz' and base_name.lower().endswith('.nii'):
        file_extension = ".nii.gz"
        
    # New data file path based on fID
    out_file_path = os.path.join(output_dir, f"{filename_id}{file_extension}") 
    
    # Copy (changing filename of course) data file if there is NO mask file
    if not mask_file_path:
        try:
            subprocess.run(["cp", data_file_path, out_file_path], check=True) # Copy the file using the 'cp' command
            print(f"Copied: {out_file_path}")
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Error occurred while copying file: {e}")     
    else:
        # Check if the mask file exists
        if not isinstance(mask_file_path, str) or not os.path.isfile(mask_file_path):
            print(f"Error: Invalid mask file path '{mask_file_path}'.")
            return

        # Apply mask file to the data file 
        # If nifti image
        if file_extension.lower() == '.nii' or file_extension.lower() == '.nii.gz':
            try:
                # load data file: NIfTI image
                img = nib.load(data_file_path)
                img_data = img.get_fdata()
                # load mask file: NIfTI image
                mask = nib.load(mask_file_path)
                mask_data = mask.get_fdata() # 1=to be included, 0=to be excluded
                # keep only the first a few dimensions of img_data to match the dimensions of mask_data
                # this is for the cases such as img_data.ndim==[97,115,97,1] & mask_data.ndim==[97,115,97]
                if img_data.ndim > mask_data.ndim:
                    # Create a slice object to select the first a few dimensions
                    slice_obj = tuple([slice(None)] * mask_data.ndim + [0] * (img_data.ndim - mask_data.ndim))
                    img_data_x = img_data[slice_obj]
                else:
                    img_data_x = img_data
                # apply the mask
                masked_data = np.where(mask_data == 1, img_data_x, np.nan)
                # create a new NIfTI image with the masked data
                masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
                # Save the output
                nib.save(masked_img, out_file_path)
                print(f"Masked output: {out_file_path}") # print information
            except Exception as e:
                print(f"Error occurred while processing NIfTI image: {e}")
        else:
            # load data file: .csv, .tsv
            try:
                data = np.genfromtxt(data_file_path, delimiter='\t')
                # load mask file: .csv, .tsv
                # TO BE DONE: Add code here for processing .csv or .tsv files
                print("Processing .csv or .tsv files not implemented yet.")
            except Exception as e:
                print(f"Error occurred while processing .csv or .tsv file: {e}")

# Function to apply a dataframe of mask to a single csv file & update this csv file
def apply_mask_to_single_csv_file(args):
    """Apply mask to a single csv file & save the new csv file with the same filename

    Args:
        csv_file (str): Path to a single csv file contains at least a column of 'Yvar'.
        mask_yvar (list): List of Yvars that are 'V'+numbers.

    Output:
        Updated CSV file that conatins elements with the same 'Yvar' in mask_yvar.
    """
    csv_file, mask_yvar = args
    df_csv = pd.read_csv(csv_file)
    df_csv[df_csv['Yvar'].isin(mask_yvar)].to_csv(csv_file, index=False)
    print(f"Updated CSV file using inclusive mask: {csv_file}.")
           
# Function to flatten a high-dimensional data & save into multiple one-row csv files
def flatten_single(args):
    """
    Flatten high-dimensional data (e.g. 2D correlation matrix & 3D nifti) into 
    a one-dimensional NumPy array and save it into multiple CSV file.

    Args:
        args (tuple): A tuple containing the data file path, the output directory, the filename ID, and the number of segments.
    
    Returns:
        A dataframe (for a single mask without num_segments): Yvar - 'V'+number; mask -- values of flattened mask:
    """
    data_file_path, output_dir, filename_id, num_segments = args

    # Check if the data file exists
    if not os.path.isfile(data_file_path):
        print(f"Warning: data file '{data_file_path}' does not exist.")
        return

    # Get the data file basename & extension
    base_name, file_extension = os.path.splitext(os.path.basename(data_file_path)) 
    
    # If nifti image
    if file_extension.lower() == '.nii' or (file_extension.lower() == '.gz' and base_name.lower().endswith('.nii')):
        # load NIfTI image
        nifti_img = nib.load(data_file_path)
        nifti_data = nifti_img.get_fdata()

        # flatten NIfTI
        flattened_data = nifti_data.flatten()
    else:
        # load .csv file
        data = np.genfromtxt(data_file_path, delimiter='\t')
        if np.isnan(data).all():
            data = np.genfromtxt(data_file_path, delimiter=',')
        
        # Check if the array is a symmetric matrix
        is_symmetric = np.allclose(data, data.T, equal_nan=True) if data.ndim == 2 and data.shape[0] == data.shape[1] else None
        
        # Symmetric matrix
        if is_symmetric:
            # Create a boolean mask for the upper triangle (without diagonal) of the matrix
            mask = np.triu(np.ones(data.shape, dtype=bool), k=1)
            # Extract the upper matrix elements
            flattened_data = data[mask]
        else:
            # Asymmetric matrix
            flattened_data = data.flatten()

    if num_segments is None:
        # Specially used for an inclusive mask (no segmentation needed)
        df = pd.DataFrame({
        'Yvar': [f'V{i}' for i in range(len(flattened_data))],
        'mask': flattened_data
         })
        # remove rows of 0 (out of mask)
        df = df[df['mask'] != 0]
        # return df
        return df
    else:
        # For all data
        # Construct the folder path (using the filename ID) to contain segmented csv files
        csv_folder_path = os.path.join(output_dir, f"{filename_id}")
        os.makedirs(csv_folder_path, exist_ok=True)

        # Save the flattened data as a CSV file
        # including reshape the data to have one row and multiple columns
        # np.savetxt(csv_file_path, flattened_data.reshape(1, -1), delimiter=",")
        
        # Calculate the size of each segment
        segment_size = len(flattened_data) // num_segments
        
        # Split the array, create filenames, and save segments using list comprehension
        [np.savetxt(
            os.path.join(csv_folder_path, f"V{i*segment_size}_{(i+1)*segment_size if i < num_segments-1 else len(flattened_data)}.csv"),
            flattened_data[i*segment_size:(i+1)*segment_size if i < num_segments-1 else len(flattened_data)].reshape(1, -1),
            delimiter=',', fmt='%g'
        ) for i in range(num_segments)]
        
        print(f"Flattened data saved in {csv_folder_path}")

# Stack (Vertically combine) segmented csv files into a new csv
def segment_stack_single(args):
    
    output_file, list_zip = args
    
    def read_csv_with_fID(single_zip):
        # Extract the parent folder name as the subject ID
        # fID = os.path.basename(os.path.dirname(file_path))
        
        # Read the CSV file
        df = pd.read_csv(single_zip[0], header=None)
        
        # Add the fID as the first column
        df.insert(0, 'fID', single_zip[1])
        
        return df

    # Use a generator expression to read CSV files with fIDs
    # df_list = (read_csv_with_fID(f) for f in list_zip)
    df_list = [read_csv_with_fID(f) for f in tqdm(list_zip, desc="Reading CSV files")]

    # Concatenate all dataframes
    combined_df = pd.concat(df_list, ignore_index=False)

    # Extract start_index and end_index from the output filename
    start_idx, end_idx = [int(s) for s in os.path.basename(output_file).replace('.csv', '').replace('V', '').split('_')]

    # Create new column names
    new_columns = ['fID'] + [f'V{i}' for i in range(start_idx, end_idx + 1)] # include the start_idx but NOT the end_idx according to Python rule

    # Rename columns, ensuring we don't exceed the number of columns in the DataFrame
    combined_df.columns = new_columns[:len(combined_df.columns)]
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # Save the combined dataframe to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to: {output_file}")


# Function to run statistical analysis by calling R script
def r_script(r_script_path, args):
    # Make sure R could be called
    # For example, endter the below code in command line before running IBMMA:
    # module load R/latest # for instance if you want to use the most updated R
    # module load /usr/local/packages/R/4.2.2/bin/R # for instance if you want to use some specific version of R
    
    # print information
    t0 = time.time() # start time
    print(f"\n-- R script executor (time consuming for large datasets) ... ")
    lambda *args: [print(arg) for arg in args]
    
    # Convert arguments to string
    str_args = [str(arg) for arg in args]

    try:
        cmd = ['Rscript', os.path.abspath(r_script_path)] + str_args # Construct the command
        subprocess.run(cmd, stdout=subprocess.PIPE) # Execute the command
    except:
        cmd = ['/usr/local/packages/R/4.2.2/bin/Rscript', os.path.abspath(r_script_path)] + str_args  # Construct the command (for Duke BIAC cluster)
        subprocess.run(cmd, stdout=subprocess.PIPE) # Execute the command
        
    # print ending info
    print(f"-- R script executor completed!\nTime elapsed (in secs): {time.time()-t0}\n")

# Function to run r_script that receives two inputs
def r_script2(args):
    arg1, file_args = args
    return r_script(arg1, file_args)

# Function to concatenate csv files
def concat_csv_single(args):
    """
    Concatenate csv files
    
    Args:
        folder_path: path to the process folder of the data type of interest
        result_dir:  path to the result  folder of the data type of interest
        model_name:  model name
        meta_mega:   "Meta" or "Mega"
        tidy_glance: "TIDY" or "GLANCE"
        subfolder1:  the immediate subfolder of TIDY & GLANCE
        subfolder2:  the immediate subfolder of subfolder1 (for TIDY only)

    Outputs:
        Save concatenated csv file into result folder.
    """
    # Arguments
    folder_path, result_dir, model_name, meta_mega, tidy_glance, subfolder1, subfolder2 = args
    
    # List all csv files of interest
    if tidy_glance == 'GLANCE':
        list_files = [os.path.join(folder_path, 'stats', segment_f, meta_mega, tidy_glance, subfolder1, model_name+'.csv')
                for segment_f in os.listdir(os.path.join(folder_path, 'stats'))
                if segment_f.startswith('V') and os.path.isdir(os.path.join(folder_path, 'stats', segment_f))]
        out_dir = os.path.join(result_dir, meta_mega, tidy_glance, subfolder1) # path to the output folder
    else:
        list_files = [os.path.join(folder_path, 'stats', segment_f, meta_mega, tidy_glance, subfolder1, subfolder2, model_name+'.csv')
                for segment_f in os.listdir(os.path.join(folder_path, 'stats'))
                if segment_f.startswith('V') and os.path.isdir(os.path.join(folder_path, 'stats', segment_f))]
        out_dir = os.path.join(result_dir, meta_mega, tidy_glance, subfolder1, subfolder2) # path to the output folder

    # Remove non-existent files
    list_files = [file for file in list_files if os.path.exists(file)]

    # Concatenate csv files
    if list_files:
        combined_csv = pd.concat([pd.read_csv(f) for f in list_files])
    
        # Sort dataframe based on 'Yvar'
        df_sorted = combined_csv.sort_values(by='Yvar',  key=lambda x: [int(''.join(char for char in str(y) if char.isdigit())) for y in x])
        
        # Save into new csv file
        [os.makedirs(out_dir, exist_ok=True) for _ in (True,) if not os.path.exists(out_dir)] # make the output folder if it does not exist
        df_sorted.to_csv(os.path.join(out_dir, model_name + '.csv'), index=False) # save into csv file with model name

# Function to correct for multiple comparisons using FDR method
def p_correct_fdr_single(file):
        """
        FDR correction for multiple comparisons.
        
        Args:
            file (str): Path to the file of uncorrected p_values.
        
        Outputs:
            save p_corrected_fdr in a new path under the directory of "p.value.fdr".
        """ 
        
        # Load csv files of p_values & turn into np.array
        df = pd.read_csv(file)
        
        # Perform FDR correction, handling potential division by zero
        mask = df['p.value'].notna() & (df['p.value'] != 0) & (df['p.value'] != 1)

        # Create a new column filled with NaN
        df['p.value.fdr'] = np.nan
        
        if mask.sum() > 0:
            reject, p_corrected, _, _ = multipletests(df.loc[mask, 'p.value'], method='fdr_bh')
            df.loc[mask, 'p.value.fdr'] = p_corrected # Assign corrected p-values only to non-NaN entries
            
            # Handle extreme p-values
            df.loc[df['p.value'] == 0, 'p.value.fdr'] = 0
            df.loc[df['p.value'] == 1, 'p.value.fdr'] = 1
        
        # Keep the columns of interest
        df1 = df[["Yvar","p.value.fdr"]]
        
        # Create the new file name
        file1 = file.replace('p.value', 'p.value.fdr')
    
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file1), exist_ok=True)
    
        # Save the updated DataFrame
        df1.to_csv(file1, index=False)

# Function to get -log(p_val)
def neg_log10_single(file):
        """
        -log10(p_val).
        
        Args:
            file (str): Path to the file of p_values.
        
        Outputs:
            save -log(p_values) in the same folder of p_values.
        """ 
        
        # Load csv files of p_values & turn into np.array
        df = pd.read_csv(file)
        colname = df.columns[1] # the column other than "Yvar"
        
        # Apply -log10 transformation to the other column
        df[colname] = -np.log10(df[colname])
        
        # Save
        parent_dir, file_name = os.path.split(file)
        file1 = os.path.join(parent_dir, "neg_log10_" + file_name)
        df.to_csv(file1, index=False)
        
# Function to reverse statistical outputs back to the orginal dimensions
def reverse_single(args):
    """
    Reverse statistical outputs (csv file) back to orginal dimension
    
    Args:
        csv_file: full path to the csv file of statistical outputs
        total_length:  the number of elements in the flattened file
        file_type: "NIFTI", "Symmetric Matrix", "CSV"
        sample_file: a sample file (masked) to provide original dimensions. Default = None

    Outputs:
        Reverse the statistical outputs to its original dimension and file type.
    """
    # Arguments
    csv_file, total_length, file_type, sample_file = args
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Find missing values
    labels = [f"V{i}" for i in range(total_length)] # labels if there is no missing value
    missing_labels = set(labels) - set(df['Yvar'].unique()) # get the labels of missing value
    
    # Fill missing labels with NaN
    if missing_labels is not None:
        # a new dataframe to contain the missing labels
        new_rows = pd.DataFrame({ 'Yvar': list(missing_labels) })
        # add NaNs to the other columns
        for col in df.columns:
            if col != 'Yvar':
                new_rows[col] = np.nan
        # append the new rows to the DataFrame
        df1 = pd.concat([df, new_rows], ignore_index=True)
        
    # Sort df1 according to labels (ascending order)
    df1['sort'] = df1['Yvar'].str.extract(r'(\d+)').astype(int)
    df1 = df1.sort_values('sort')
    df1 = df1.drop('sort', axis=1)
    
    # Reverse to the original dimensions
    if file_type == 'NIFTI':
        parent_path = os.path.dirname(csv_file) # Get the parent directory
        basename, _ = os.path.splitext(os.path.basename(csv_file)) # Get the basename (filename)
        
        # load sample nifti image (in fact, its affine: image.affine)
        image = nib.load(sample_file) # use the info of this image cause all images are of the same shape
        # load image data (in fact, its shape: data.shape, e.g., (97,115,97))
        data = image.get_fdata()
        # get file extension
        base_name, file_extension = os.path.splitext(sample_file) # get file extension
        if file_extension.lower() == '.gz' and base_name.lower().endswith('.nii'):
            file_extension = ".nii.gz"
        
        # reverse the data according to its original dimensions & affine
        reversed_data = df1.iloc[:,1].values.reshape(data.shape) # reshape according to original nifti domensions
        reversed_image = nib.Nifti1Image(reversed_data, image.affine) # make image according to original nifti affine
        
        # save nifti image
        nib.save(reversed_image, os.path.join(parent_path, 'OUT_'+basename+file_extension))
    
    elif file_type == 'Symmetric Matrix':
        # Calculate the size of the matrix
        n = int((-1 + np.sqrt(1 + 8 * total_length)) / 2) + 1

        # Create an empty matrix filled with NaN
        matrix = np.full((n, n), np.nan)
        # Fill the upper triangle (without diagonal) with elements from the input array
        matrix[np.triu_indices(n, k=1)] = df1.iloc[:,1]
        
        # Save matrix
        parent_path = os.path.dirname(csv_file) # Get the parent directory
        basename = os.path.basename(csv_file) # Get the basename (filename)
        np.savetxt(os.path.join(parent_path, 'OUT_'+basename), matrix, delimiter=",") # Save the matrix to a CSV file
    
        # Plot the connectivity matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        abs_limit = max(abs(np.nanmax(matrix)),abs(np.nanmin(matrix))) # limits of values
        plotting.plot_matrix(matrix, title='', figure=fig,
                            labels=[f'{i+1}' for i in range(n)],
                            vmax=abs_limit, vmin=-abs_limit, reorder=False)

        fig.savefig(os.path.join(parent_path, 'FIG_'+basename+'.png'), dpi=300, bbox_inches='tight') # Save the plot to a PNG file
    
    else:
        print(f'Reverse problem: unknown data type, not .nii/.nii.gz, nor matrix\n')


# Function to make roi results report
def roi_results(model_name, csv_files, my_rois):
    """
    Combine the seperated results files into one per model
    
    Args:
        model_name (strings):           Model name, e.g., "model_01".
        csv_files (list of strings):    A list of paths to the csv files of statistical outputs (TIDY & GLANCE).
        my_roi_file (dataframe):        Dataframe of ROI defination.

    Outputs:
        One .xlsx file per model, and one sheet per effect.
    """
    
    # Labels of all ROIs
    label_rois = my_rois.Label.tolist()
    
    # Create dictionary using dictionary comprehension
    dict_labels = {'V'+str(index): value for index, value in enumerate(label_rois)}
    
    # Files of TIDY and GLANCE
    TIDY_files   = [file for file in csv_files if os.path.basename(file).startswith(model_name) and file.endswith('.csv') and os.path.normpath(file).split(os.sep)[-4]=='TIDY'] 
    GLANCE_files = [file for file in csv_files if os.path.basename(file).startswith(model_name) and file.endswith('.csv') and os.path.normpath(file).split(os.sep)[-3]=='GLANCE']
    
    # Output file path
    common_path = os.path.commonpath(csv_files) # Get the common parts of the paths
    out_file = os.path.join(common_path, 'ROIs_Results_'+model_name+'.xlsx')

    # GLANCE: Loop through the files and read each into a dataframe
    df_GLANCE = pd.DataFrame() # Initialize an empty dataframe
    for file in GLANCE_files:
        df = pd.read_csv(file)
        if df_GLANCE.empty:
            df_GLANCE = df
        else:
            # Merge on the common variable 'Yvar'
            df_GLANCE = pd.merge(df_GLANCE, df, on='Yvar')
    
    # TIDY: Loop through the files and read each into a dataframe
    # # Unique effects
    # effects = set(os.path.basename(os.path.dirname(file_path)) for file_path in TIDY_files)
    # Group csv files by parent folder (TIDY effects)
    folders = {} # dictionary, folder name: files within this folder
    for csv_file in TIDY_files:
        parent_folder = os.path.basename(os.path.dirname(csv_file)) # Get the 1st parent folder name
        if parent_folder not in folders:
            folders[parent_folder] = []
        folders[parent_folder].append(csv_file)

    # Merge csv files in the same folder into one dataframe
    with pd.ExcelWriter(out_file) as writer:
        for folder_name, ffiles in folders.items():
            df_list = [pd.read_csv(file) for file in ffiles] # Get a list of dataframes, one per TIDY output
            df_TIDY = pd.DataFrame() # Initialize an empty dataframe
            df_TIDY = pd.concat(df_list).groupby('Yvar').first().reset_index() # Merge the list of dataframes using the common variable "Yvar"
            df_TIDY = df_TIDY.merge(df_GLANCE, on='Yvar', how='inner') # merge TIDY & GLANCE
            df_TIDY['Yvar'] = df_TIDY['Yvar'].replace(dict_labels) # Replace Yvar with ROI labels
            df_TIDY = df_TIDY.sort_values(by='p.value', ascending=True)
            # Save 
            if not df_TIDY.empty:
                df_TIDY.to_excel(writer, sheet_name=folder_name, index=False)
                print(f'ROI statistical results saved in sheet={folder_name} in file={out_file}\n')
            else:
                print(f'No data for {folder_name}, skipping...\n')

           
        
        
# Class for Mega analysis
class Mega:
    def __init__(self, num_processes=None):
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count() # use all CPUs if not specified

    def mask(self, subjects_csv_path, col_name, col_mask_name, process_dir):
        """
        Mask data files so that values out of mask are converted to NaN.
        
        Args:
            subjects_csv_path (str): Path to the CSV file containing the data file paths and fIDs.
            col_name: Name of the column that stroes the data file paths.
            col_mask_name: Name of the column that stores the mask file paths.
            process_dir (str): Path to the directory where the MEGA analysis should take.
        
        Outputs:
            save masked data per file.
        """
        # print information
        t0 = time.time() # start time
        print(f"\nMask data files (time consuming for multiple files) ... ")
        
        # output_dir
        output_dir = os.path.join(process_dir,'masked')
        # make the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read the data file paths and filename IDs from the CSV file
        subjects_df = pd.read_csv(subjects_csv_path)
        # remove rows that are empty in the columns of fID & col_name
        subjects_df.dropna(subset=['fID', col_name], inplace=True)

        # Aarguments
        my_args = [(row[col_name], 
                    row[col_mask_name] if col_mask_name in row.index else None, 
                    output_dir, 
                    row['fID']) 
                   for _, row in subjects_df.iterrows()]
        
        # # For test purpose only !!
        # mask_single(my_args[0])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(mask_single, my_args)
            
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
        print(f"Mask data files completed!\nTime elapsed (in secs): {time.time()-t0}\n")

    def flatten(self, subjects_csv_path, col_name, process_dir, num_segments):
        """
        Flatten any dimensions of data into multiple segmented 1D data.
        
        Args:
            subjects_csv_path (str): Path to the CSV file containing the data file paths and fIDs.
            col_name (str): Name of the column that stroes the data file paths.
            process_dir (str): Path to the directory where the MEGA analysis should take.
            num_segments (num): Number of segments
        
        Outputs:
            save flattened data files into folders per fID.
        """
        # print information
        t0 = time.time() # start time
        print(f"\nFlatten data files into csv files (time consuming for multiple files) ... ")
        
        # output_dir
        output_dir = os.path.join(process_dir,'flattened')
        # make the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read the data file paths and filename IDs from the CSV file
        subjects_df = pd.read_csv(subjects_csv_path, na_values='NaN')
        # remove rows that are empty in the columns of fID & col_name
        subjects_df.dropna(subset=['fID', col_name], inplace=True)

        # Aarguments
        my_args = [(row[col_name], output_dir, row['fID'], num_segments) for _, row in subjects_df.iterrows()]
        
        # # For test purpose only !!
        # flatten_single(my_args[0])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(flatten_single, my_args)
            
        # print ending info
        print(f"Flatten & segment data completed!\nTime elapsed (in secs): {time.time()-t0}\n")

    def segment(self, process_dir):
        """
        Vertically combine the same sgmentation across subjects into a new file.
        
        Args:
            process_dir (str): Path to the processing directory for the corresponding data pattern.
        
        Outputs:
            save combined segmented data files.
        """
        # Print information
        t0 = time.time() # start time
        print(f"\nCombine flattened & segmented data to save into new csv files (time consuming for big datasets) ... ")
        
        # List of fID
        list_fID = os.listdir(os.path.join(process_dir, 'flattened'))
        # List of filenames of segments (based on the folder of the 1st fID)
        list_segments = os.listdir(os.path.join(process_dir, 'flattened',list_fID[0]))
        
        # csv_files = [os.path.join(process_dir,'flattened',folder,list_segments[0]) for folder in list_fID]
        # List all csv files in a given subject's folder of flattened data files
        # csv_files = [os.path.join(process_dir, 'flattened', folder, list_segments[0]) 
        #     for folder in list_fID 
        #     if not folder.startswith('.') and list_segments[0] and not list_segments[0].startswith('.')]
        
        # my_args = [
        #     (
        #         os.path.join(process_dir, 'segmented', segment),
        #         list(zip(
        #             [os.path.join(process_dir, 'flattened', folder, segment) for folder in list_fID],
        #             list_fID
        #         ))
        #     )
        #     for segment in list_segments
        # ]
        my_args = [
            (
                os.path.join(process_dir, 'segmented', segment),
                list(zip(
                    [os.path.join(process_dir, 'flattened', folder, segment) 
                        for folder in list_fID 
                        if not folder.startswith('.') and not segment.startswith('.')],
                    [folder for folder in list_fID if not folder.startswith('.')]
                ))
            )
            for segment in list_segments 
            if not segment.startswith('.')
            ]
        
        # # For test purpose only !!
        # segment_stack_single(my_args[0])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes//2) as pool:
            pool.map(segment_stack_single, my_args)
            
        # print ending info
        print(f"Stack segmented data completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
    def stat(self, folder_path, output_dir, R_script_path, subjects_csv_path, model_name, model_txt):
        """
        Statistical analysis across all segmented data files
        
        Args:
            folder_path (str): Path to the folder of the target data type.
            output_dir (str): Path to the folder of statistical outputs.
            R_script_path (str): Path to the R script for statistical modelling.
            subjects_csv_path (str): path to Subjects.csv
            model_name (str): name of the model, e.g., 'model_01'
            model_txt (str): text of the model's formula, e.g., 'lm(Yvar ~ GROUP + AGE + SEX)'
        
        Outputs:
            save statistical outputs (TIDY & GLANCE).
        """  
        # path to the folder of segmented data
        folder_path = os.path.abspath(folder_path) # use absolute path to avoid errors
        segmented_path = os.path.join(folder_path, 'segmented') 
        
        # list all csv files within the folder of segmented
        segmented_files = [os.path.join(segmented_path, f) for f in os.listdir(segmented_path) if f.endswith('.csv')]
           
        # Arguments
        # --- arg1, full path to xfile.csv, which contains predictors (row = observation, column = variable).
        # --- arg2, full path to yfile.csv, which contains outcome variables (row = observation, column = variable).
        # --- arg3, full path to the output directory.
        # --- arg4, model name, e.g. "model_01"
        # --- arg5, texts of model formula, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))" or "lm(Yvar ~ GROUP + AGE + SEX)".
        my_args = [(os.path.abspath(subjects_csv_path), 
                    file, 
                    os.path.join(output_dir, os.path.basename(file)[:-len('.csv')], 'Mega'),
                    model_name,
                    model_txt) 
                   for file in segmented_files]
        
        # # for test purpose only
        # r_script(R_script_path, my_args[0])
        
        t0 = time.time()  # Record the start time

        # Statisyical analysis across all segments in sequence
        for my_arg in tqdm(my_args, desc="Statistical Modelling", miniters=1):
            r_script(R_script_path, my_arg)  # statistical analysis (parallel processing) using R script
        
        # Print ending info
        print(f"Statistical analyses completed!\nTime elapsed (in secs): {time.time()-t0}\n")

    def concatenate(self, folder_path, result_dir, model_name):
        """
        Concatenate CSV files of statistical outputs across all segments per model.
        
        Args:
            folder_path (str): Path to the folder of the target data type.
            result_dir (str):  Path to the folder of statistical outputs of the target data type.
            model_name (str):  Name of the model, e.g., 'model_01'
        
        Outputs:
            save concatenated statistical outputs (TIDY & GLANCE).
        """ 
        # print information
        t0 = time.time() # start time
        print(f"\nConcatenating CSV files ... ")
        
        # List all subfolder names under GLANCE & TIDY
        subfolders_GLANCE = list({os.path.basename(path) for path, _, _ in os.walk(folder_path) if os.path.join('Mega','GLANCE',os.path.basename(path)) in path}) # all subfolder names of GLANCE
        subfolders_TIDY0  = list({os.path.basename(path) for path, _, _ in os.walk(folder_path) if os.path.join('Mega','TIDY')                          in path and os.path.basename(path) != 'TIDY'}) # all subfolder names of TIDY
        subfolders_TIDY1  = list({os.path.basename(path) for path, _, _ in os.walk(folder_path) if os.path.join('Mega','TIDY',os.path.basename(path))   in path}) # immediately subfolder names of TIDY
        subfolders_TIDY2  = list(set(subfolders_TIDY0) - set(subfolders_TIDY1))
        
        # Arguments for GLANCE & TIDY
        args_GLANCE = [(folder_path, result_dir, model_name, 'Mega', 'GLANCE', subfolder1, None)
                   for subfolder1 in subfolders_GLANCE]
        args_TIDY   = [(folder_path, result_dir, model_name, 'Mega', 'TIDY',   subfolder1, subfolder2)
                   for subfolder1 in subfolders_TIDY1
                   for subfolder2 in subfolders_TIDY2]
        my_args = args_GLANCE + args_TIDY # combine the lists of tuples
        
        # # For test purpose only !!
        # concat_csv_single(my_args[9]) 
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(concat_csv_single, my_args) 
 
        # print ending info
        print(f"Concatenating CSV files completed!\nTime elapsed (in secs): {time.time()-t0}\n")

    def reverse(self, process_dir, result_dir, model_name, mask1, path_R_pTFCE, my_rois_path):
        """
        (1) Apply inclusive mask (if available) to restrict all statistical outputs within the mask
        (2) FDR correction (default) for p-values; 
        (3) Negatively log10 transformed p-values; 
        (4) Reverse the concatenate CSV files of statistical outputs back to its original dimensions;
        (5) pTFCE for .nii and .nii.gz.
        
        Args:
            process_dir (str):  Path to the folder of the target data type in the Process folder.
            result_dir (str):   Path to the folder of statistical outputs of the target data type.
            model_name (str):   Name of the model, e.g., 'model_01'.
            mask1 (str):        Path to the whole brain inclusive mask image, e.g., '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/brain_mask.nii'.
            path_R_pTFCE (str): Path to the R script for pTFCE on .nii and .nii.gz.
            my_rois_path (str): Path to the file of my_rois, e.g., 'MY_ROIs.xlsx'.
        
        Outputs:
            save reversed concatenated statistical outputs (TIDY & GLANCE).
        """ 
        ## (1) Inclusive mask (if available)
        
        # Print information
        t0 = time.time() # start time
        print(f"\nInclusive mask: {mask1}")
        
        # Flatten MASK1 if it is available
        if mask1 is None:
            pass
        else: 
            # flatten the mask1 & save into mask1.csv
            df_mask = flatten_single((mask1,result_dir,'mask1',None))
            # apply mask to all csv files & update the csv files
            # apply_mask_to_csv_files(df_mask, result_dir)
                # List all CSV files recursively
            csv_files = [os.path.join(root, file) 
                        for root, _, files in os.walk(result_dir) 
                        for file in files if file.endswith('.csv')]
            
            # Prepare arguments for multiprocessing
            args = [(csv_file, df_mask['Yvar'].tolist()) for csv_file in csv_files]
            
            # For test purpose only
            apply_mask_to_single_csv_file(args[0])
            
            # Use parallel processing
            with mp.Pool(processes=self.num_processes) as pool:
                processed_files = pool.map(apply_mask_to_single_csv_file, args)
            
            print(f"Updated {len(processed_files)} CSV files using inclusive mask.")

        
        ## (2) FDR correction (default)
        
        # Print information
        t0 = time.time() # start time
        print(f"\nFDR correction (default) ... ")
        
        # List all CSV files (model_name + ".csv") recursively under the folder of "p.value"
        csv_files = [os.path.join(root, file)
             for root, _, files in os.walk(result_dir)
             for file in files
             if file == (model_name + ".csv") and os.path.normpath(root).split(os.path.sep)[-2] == "p.value"]
        
        # # For test purpose only !!
        # p_correct_fdr(csv_files[0]) 
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(p_correct_fdr_single, csv_files) 
 
        # print ending info
        print(f"FDR correction completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
        
        ## (3) Negatively log10 transformed p-values
        
        # Print information
        t0 = time.time() # start time
        print(f"\nNegatively log10 transformation ... ")
        
        # List all CSV files (model_name + ".csv") recursively under the folders whose name starts with "p.value"
        csv_files = [os.path.join(root, file)
             for root, _, files in os.walk(result_dir)
             for file in files
             if file == (model_name + ".csv") and os.path.normpath(root).split(os.path.sep)[-2].startswith("p.value")]
        
        # # For test purpose only !!
        # neg_log10_single(csv_files[0]) 
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(neg_log10_single, csv_files) 
 
        # print ending info
        print(f"Negatively log10 transformation completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
        ## (4) Reverse
        # Print information
        t0 = time.time() # start time
        print(f"\nReverse statistical outputs back to original dimensions ... ")
        
        # List all CSV files recursively in TIDY & GLANCE folders of the given model
        csv_files = [os.path.join(root, file)
             for root, _, files in os.walk(result_dir)
             for file in files
             if file == (model_name + ".csv") or file == ("neg_log10_" + model_name + ".csv")]
        
        # Length of flattened but not segmented data
        numbers = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(os.path.join(process_dir, 'segmented'))
           if f.startswith('V') and f.endswith('.csv') and '_' in f and f.split('_')[1].split('.')[0].isdigit()]
        total_length = max(numbers)
        
        # Info of the input data & flattened data
        f_data= [os.path.join(process_dir, 'masked', f) for f in os.listdir(os.path.join(process_dir, 'masked')) if os.path.isfile(os.path.join(process_dir, 'masked', f))]
        base_name, file_extension = os.path.splitext(f_data[0]) # get file extension
        # Revise file extension if it is ".nii.gz"
        if file_extension.lower() == '.gz' and base_name.lower().endswith('.nii'):
            file_extension = ".nii.gz"
        
        # File type of data
        if file_extension in ['.nii', '.nii.gz']:
            file_type = 'NIFTI' # NIFTI images
            sample_file = f_data[0]
        else:
            # Check if the array is a symmetric matrix
            data = np.genfromtxt(f_data[0], delimiter=',') # load the 1st .csv file
            # is_symmetric = np.allclose(data, data.T, equal_nan=True)
            is_symmetric = np.allclose(data, data.T, equal_nan=True) if data.ndim == 2 and data.shape[0] == data.shape[1] else None
            file_type = 'Symmetric Matrix' if is_symmetric else 'CSV' # file type for .csv files
            sample_file = None

        # Arguments
        my_args = [(csv_file, total_length, file_type, sample_file)
                   for csv_file in csv_files]
        
        # # For test purpose only !!
        # reverse_single(my_args[10])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(reverse_single, my_args) 
            
        # Print ending info
        print(f"Reverse statistical outputs back to original dimensions completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
        
        ## (5) pTFCE
        # Print information
        t0 = time.time() # start time
        print(f"\npTFCE for .nii and .nii.gz ... ")
        
        # Folder of statistic
        folder_statistic = os.path.join(result_dir,'Mega','TIDY','statistic')
        
        # List all .nii and .nii.gz recursively in TIFY/statistic folder of the given model
        my_args = [
            (
                os.path.join(root, file),
                os.path.join(root, file).replace("statistic", "df"),
                mask1
            )
            for root, dirs, files in os.walk(folder_statistic)
            for file in files
            if file.startswith("OUT_" + model_name) and (file.endswith(".nii") or file.endswith(".nii.gz"))
        ]
        
        # Parallel processing
        # Prepare arguments for multiprocessing
        pool_args = [(path_R_pTFCE, arg) for arg in my_args]
        
        # # # For test purpose only !!
        # r_script2(pool_args[0])

        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(r_script2, pool_args)
            
        # Print ending info
        print(f"Reverse statistical outputs back to original dimensions completed!\nTime elapsed (in secs): {time.time()-t0}\n")

        
        # # Make results report (one .xlsx file per model, one sheet per effect)
        # # Check if the data in the masked folder is 1D (i.e., ROI-based data)
        # data = np.genfromtxt(f_data[0], delimiter=',') # load the 1st .csv file
        # if data.ndim == 1:
        #     my_rois = pd.read_excel(my_rois_path, sheet_name='MY_ROIs', dtype='object')
        #     roi_results(model_name, csv_files, my_rois)
        
        # # Print ending info
        # print(f"ROI results report completed!\nTime elapsed (in secs): {time.time()-t0}\n")
 