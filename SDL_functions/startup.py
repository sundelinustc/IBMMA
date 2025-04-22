# (0) Packages
import os  # For getting and setting paths
import time  # For measuring elapsed time
import pandas as pd  # For working with DataFrames
import numpy as np
from multiprocessing import Pool  # For parallel processing
from functools import wraps  # For using decorators


def func_list_files(args):
    """
    Function to list the files with a given string pattern in a folder
    Note: this function CAN NOT be incorporated into the class of Clean() to avoid conflicts

    Args:
        folder (str): Path to the folder in which to search for files of interest
        pattern (str): String pattern that all files of interest have in their file names, e.g., "task-rest_falff.nii.gz"
        data_type (str): Data type, e.g., "fALFF", "ReHo"
        data_attr (str): Data attribute, e.g., "data", "info", "mask"

    Returns:
        df (pandas.DataFrame): A DataFrame with 3 columns:
            full_path - Full path to the file of interest
            fID - Concatenation of site name and subject ID, separated by "_"
            data_name - Concatenation of data_attr + "_" + data_type, e.g., "data_fALFF_alff" and "mask_reHo"
    """
    folder, pattern, data_type, data_attr = args
    data = {'full_path': [], 'fID': [], 'data_name': []}
    
    # go through all subfolders recrusively
    for root, dirs, files in os.walk(folder):
        for file in files:
            # if the pattern is a part of the file's name
            if pattern in file:
                full_path = os.path.join(root, file)
                data['full_path'].append(full_path)
                
                upper_folder = os.path.basename(os.path.dirname(full_path))
                second_upper_folder = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                third_upper_folder  = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
                fourth_upper_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(full_path)))))
                
                file_start = file.split("_")[0] # beginning of a file, e.g., "sub-61" of "sub-61_T1_n4_mni_seg_post_inbverse.nii.gz"
                
                # fID: sitename + "_" + subjectID, based on data filename
                if pattern == '_n4_mni_seg_post_inverse.nii.gz':
                    # for cerebellum acapulco data, e.g., enigma_Groningen/acapulco/output/61/sub-61_T1_n4_mni_seg_post_inverse.nii.gz
                    fID = fourth_upper_folder + "_" + upper_folder
                elif upper_folder == file_start:
                    # for new_halfpipe data, e.g., Duke/sub-1234/sub-1234_task-rest_falff.nii.gz
                    fID = second_upper_folder + "_" + file_start
                else:
                    # for CENC data, e.g., CENC1/sub-1234_task-rest_falff.nii.gz
                    fID = upper_folder + "_" + file_start
                
                data['fID'].append(fID)
                data['data_name'].append(data_attr + "_" + data_type)
    
    df = pd.DataFrame(data)
    return df

# (1) Class to load and clean important info from an Excel file
class Clean:
    def __init__(self, file_path='path_para.xlsx'):
        self.file_path = file_path # absolute or relative path to the excel file of paths & parameters
        self.project_path = os.path.dirname(os.path.abspath(self.file_path)) # its absolute path is also project path
        self.excel_file = pd.ExcelFile(self.file_path)
        self.df_all_sheets = self.read_all_sheets() # read all sheets from the excel file of paths & parameters
        pass

    # Decorator function to calculate time elapsed
    def timeit(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"-Function {func.__name__} took {end_time - start_time:.3f} secs")
            return result
        return wrapper
       
    @timeit
    def read_all_sheets(self):
        """
        Read data from all sheets in the Excel file and return a dictionary of DataFrames.

        Returns a dictionary where the keys are sheet names, and the values are DataFrames
        containing the data from each sheet.
        """
        sheet_dfs = {} # initialize a dictionary
        with pd.ExcelFile(self.file_path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read the sheet as a DataFrame of strings
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name, dtype='object')
                    
                    # Filter the DataFrame to keep only rows where 'excluded' is 0
                    df = df[df['EXCLUDED'] == 0]
                    
                    # Add the filtered DataFrame to the dictionary
                    sheet_dfs[sheet_name] = df
                except ValueError as e:
                    print(f"Error reading sheet '{sheet_name}': {e}")

        return sheet_dfs

    @timeit
    def print_sheets(self, sheet_dfs):
        """
        Print each DataFrame from the given dictionary of DataFrames.
        """
        for sheet_name, df in sheet_dfs.items():
            print(f"\nSheet: {sheet_name}")
            print(df)

    @timeit
    def preprocess_data(self):
        """
        Load data from Excel files and sheets specified in the 'demographic_clinical' sheet,
        and preprocess the data based on the information in the 'predictors' DataFrame.
        """
        data_df = self._load_from_demographic_clinical()  # Read demographic & clinical data
        # predictors_df = self._read_sheet('predictors')  # Read sheet of predictors
        predictors_df = self.df_all_sheets['predictors']

        preprocessed_df = data_df.copy()  # Make a copy of demographic & clinical data

        # Preprocess data if the 'predictors' sheet is not empty
        if not predictors_df.empty:
            # Iterate across all rows where 'excluded' is 0
            for _, row in predictors_df[predictors_df['EXCLUDED'] == 0].iterrows():
                var = row['VAR']  # Variable name used in analysis
                name = row['NAME']  # Variable name listed in demographic & clinical data
                old2new = row['OLD2NEW']  # Strings for name to var conversion: name1:var1, name2:var2,...
                values = row['VALUES']  # Values to be kept for analysis

                # Change variable names if different in analysis and raw data
                if var != name:
                    preprocessed_df = preprocessed_df.rename(columns={name: var})

                # Change variable values if specified (old1:new1, old2:new2, ...)
                if pd.notna(old2new):
                    mapping = dict(item.split(":") for item in old2new.split(","))
                    preprocessed_df[var] = preprocessed_df[var].replace(mapping)

                # Keep specified variable values (in strings)
                if pd.notna(values):
                    values_list = values.split(',') if isinstance(values, str) else values
                    preprocessed_df[var] = preprocessed_df[var].apply(lambda x: x if str(x) in values_list else pd.NA)

            # drop rows that are NaN across columns
            preprocessed_df = preprocessed_df.dropna(how='all')
            
            # set 'fID' as index
            preprocessed_df.set_index('fID', inplace=True)
                
        return preprocessed_df

    def _load_from_demographic_clinical(self):
        """
        Internal function to load data from Excel files and sheets specified in the 'demographic_clinical' sheet.
        Returns a DataFrame containing the concatenated data from all specified files and sheets.
        """
        dfs = []  # Initialize an empty list to store DataFrames
        demographic_clinical_df = self.df_all_sheets['demographic_clinical']  # Get files' paths & sheets

        # Read data if there is at least one file for demographic & clinical info
        if not demographic_clinical_df.empty:
            # Read across rows (multiple files & sheets)
            for _, row in demographic_clinical_df.iterrows():
                sheet_name = row['SHEET']
                excluded = row['EXCLUDED']
                file_paths = str(row['FILE']).split(';')  # Split file paths by semicolon

                if pd.notna(sheet_name) and excluded == 0:
                    for file_path in file_paths:
                        if pd.notna(file_path):
                            try:
                                # Read the Excel file of demographic & clinical info into a DataFrame of strings
                                data_df = pd.read_excel(file_path, sheet_name=sheet_name, dtype='object')
                                dfs.append(data_df)
                            except Exception as e:
                                print(f"Error reading file '{file_path}', sheet '{sheet_name}': {e}")

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    @timeit
    def match_files(self):
        """
        Function to search for the files in parallel

        Args:
            folder (str): Path to the folder in which to search for files of interest
            pattern (str): String pattern that all files of interest have in their file names, e.g., "task-rest_falff.nii.gz"
            data_type (str): Data type, e.g., "fALFF", "ReHo"
            data_attr (str): Data attribute, e.g., "data", "info", "mask"

        Returns:
            df (pandas.DataFrame): A DataFrame with 3 columns:
                full_path - Full path to the file of interest
                fID - Concatenation of site name and subject ID, separated by "_"
                data_name - Concatenation of data_attr + "_" + data_type, e.g., "data_fALFF_alff" and "mask_reHo"
        """
        # The sheet of data_pattern
        data_pattern_df = self.df_all_sheets['data_pattern']
        # keep rows where 'excluded' column are 0
        data_pattern_df = data_pattern_df[data_pattern_df['EXCLUDED'] == 0]  
        # remove the 'excluded' column
        # data_pattern_df.drop('EXCLUDED', axis=1, inplace=True)  

        # Parallel searching for all files of interest
        print(f"\n-----------\nSearching for all files of interest (time consuming, ~10min !!)...")
        # arguments: (folder, pattern, data_type, data_attr) tuples for parallel processing
        folder_paths = self.df_all_sheets['data_path']['PATH'].tolist() # paths to all data
        my_args0 = [
            (str(folder_path), str(row[col]), row['NAME'], col)
            for folder_path in folder_paths
            for col in data_pattern_df.columns.tolist() if col not in ['NAME','CALCULATION','EXCLUDED']
            for __, row in data_pattern_df.iterrows() if pd.notna(row[col])
        ]
        
        # two types of arguments: is the 2nd element the fullpath to an existing file?
        my_args  = [tup for tup in my_args0 if not os.path.exists(tup[1])] # the answer is NO
        my_args1 = [tup for tup in my_args0 if     os.path.exists(tup[1])] # the answer is YES
               
        # use all of the CPUs to run parallel processing
        # df = func_list_files(my_args[5]) # for test purpose only
        with Pool() as pool:
            dfs = pool.map(func_list_files, my_args)
        df = pd.concat(dfs)
        
        # Convert to a wide dataframe
        df_wide = df.pivot_table(index='fID', columns='data_name', values='full_path', aggfunc='first')
        df_wide.columns = ['FULL_PATH_' + col for col in df_wide.columns] # Add prefix to the column names
        
        # Merge demographic/clinical info (df0) with data paths (df)
        df0 = self.preprocess_data()
        df_merged0 = df0.merge(df_wide, left_index=True, right_index=True)
        
        # For columns that also start with 'fID', e.g., "fID_cerebellum"
        fid_cols = [col for col in df0.columns if col.startswith('fID')] # find these columns
        if fid_cols:        
            # merge the dataframe related to these columns with df_wide by the corresponding column that starts with "fID", respectively
            df_mergeds = [df0.reset_index().set_index(col).rename(columns={'index': 'fID'}).merge(df_wide, left_index=True, right_index=True) for col in fid_cols]
            # set 'fID' as the index for all DataFrames in df_mergeds
            df_mergeds = [df.set_index('fID') for df in df_mergeds]
            # fill NaN values in df_final with non-NaN values from df
            df_final = df_mergeds[0] # Start with the first DataFrame in df_mergeds
            for df in df_mergeds[1:]:
                df_final = df_final.combine_first(df)
            # merge df_merged0 with df_final
            df_merged = df_merged0.combine_first(df_final)
        else:
            df_merged = df_merged0
        
        # Add columns of files with fullpath, e.g., '/tmp/isilon/morey/lab/dusom_morey/Aurelio_preproc/atlases/tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.nii.gz'
        if my_args1:
            new_list = list(set([('FULL_PATH_' + tup[3] + '_' + tup[2], tup[1]) for tup in my_args1])) # keep the unique values
            for col_name, col_value in new_list:
                df_merged[col_name] = col_value
        
        # Drop the rows that are NaN across columns starts with "FULL_PATH_"
        cols = [col for col in df_merged.columns if col.startswith('FULL_PATH_')]
        df_merged = df_merged.dropna(subset=cols, how='all')
    
        # Save into Subjects.csv in the Processes folder
        path_processes = os.path.join(self.project_path, "Processes")
        # make the Processes folder if it does not exist
        os.makedirs(path_processes, exist_ok=True) 
        # save into Subjects.csv file in the Processes folder
        df_merged.to_csv(os.path.join(path_processes,'Subjects.csv'), index=True)

# # for test purpose only
# if __name__ == "__main__": 
#     print('\n\n\n\nAnything wrong here?')
#     pass
