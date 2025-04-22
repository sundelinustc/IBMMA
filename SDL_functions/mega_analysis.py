import os  # for file operations
import re # for regular expressions to match and manipulate strings of text
import sys
import numpy as np # for manipulating arrays
import pandas as pd # for working with csv files
import scipy.stats as stats # for Cohen's D and CI
import nibabel as nib # for reading and writing nifti images
import nilearn.plotting as plotting # for plotting matrix of connection
import matplotlib.pyplot as plt # for making the frame of plotting
import multiprocessing as mp # for parallel processing
import subprocess # for calling functions through OS system
import time # for measuring time elapsed
from tqdm import tqdm # for progress bar
from statsmodels.stats.multitest import multipletests # for correction of multiple comparisons
from tableone import TableOne

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
    # try:
    csv_file, mask_yvar = args
    df_csv = pd.read_csv(csv_file)
    df_csv[df_csv['Yvar'].isin(mask_yvar)].to_csv(csv_file, index=False)
    print(f"Updated CSV file using inclusive mask: {csv_file}.")
    # except Exception as e:
    #     print(f"Error applying mask or saving file {csv_file}: {str(e)}")
    #     return
           
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

# Function to filter rows of interest in the dataframe from Subjects.csv
import pandas as pd
import re
import sys

def filter_dataframe(df, filter_string=None):
    """
    Filter a pandas DataFrame using a text-based filter condition with flexible syntax.
    
    The filter string uses the following syntax:
    - Semicolons (;) or "AND" represent logical AND operations
    - Commas (,) or "OR" represent logical OR operations
    - Tilde (~) or "NOT" represent logical NOT operations
    - Parentheses () can be used to control the order of operations
    - Each condition can use comparison operators: ==, !=, <, >, <=, >=
    - Range conditions like "20 < Age < 30" are supported
    
    If filter_string is None or an empty string, returns the original DataFrame.
    
    Examples:
    - "20 < Age < 60; Sex == Female; Gender == 1; Site == Duke, Emory; Sev == 0,1"
    - "20 < Age < 30 OR Age > 60 AND Sex == Female"
    - "NOT (GROUP==0 AND AGE<=10)"
    - "~(Age<18) AND Site == Duke OR Emory"
    - "(Site == Duke OR Site == Emory) AND Age > 40"
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter
    filter_string : str, optional
        The filter condition as a text string
    
    Returns:
    --------
    pandas.DataFrame
        The filtered DataFrame
    """
    # If filter_string is None or empty, return the original DataFrame
    if not filter_string:
        return df
    
    # Set a higher recursion limit if needed but not too high
    original_recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(2000)  # Increase but not too much
        
        # Pre-process the filter string
        filter_string = preprocess_filter(filter_string)
        
        # Process the filter string using iteration instead of deep recursion
        final_mask = parse_filter(filter_string, df)
        
        # Return the filtered DataFrame
        return df[final_mask]
    except Exception as e:
        print(f"Error processing filter: {str(e)}")
        # Return the original DataFrame in case of error
        return df
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(original_recursion_limit)


def preprocess_filter(filter_string):
    """Normalize logical operators to standard form."""
    # Replace logical operators with standardized symbols
    # NOT -> ~
    filter_string = re.sub(r'\bNOT\b', '~', filter_string, flags=re.IGNORECASE)
    
    # Replace AND with ; (being careful with strings)
    result = ""
    in_quotes = False
    i = 0
    
    while i < len(filter_string):
        if filter_string[i] in ['"', "'"]:
            in_quotes = not in_quotes
            result += filter_string[i]
            i += 1
        elif not in_quotes and i + 2 < len(filter_string) and filter_string[i:i+3].upper() == 'AND':
            # Check if it's surrounded by spaces or is at the beginning/end
            if (i == 0 or filter_string[i-1].isspace()) and (i+3 >= len(filter_string) or filter_string[i+3].isspace()):
                result += ";"
                i += 3
            else:
                result += filter_string[i]
                i += 1
        elif not in_quotes and i + 1 < len(filter_string) and filter_string[i:i+2].upper() == 'OR':
            # Check if it's surrounded by spaces or is at the beginning/end
            if (i == 0 or filter_string[i-1].isspace()) and (i+2 >= len(filter_string) or filter_string[i+2].isspace()):
                result += ","
                i += 2
            else:
                result += filter_string[i]
                i += 1
        else:
            result += filter_string[i]
            i += 1
    
    return result.strip()

def parse_filter(filter_string, df, depth=0):
    """
    Parse the filter string using an iterative approach to avoid excessive recursion.
    """
    # Check recursion depth
    if depth > 50:  # Set a reasonable limit
        raise RecursionError("Filter parsing reached maximum allowed depth")
    
    # Base case: empty string
    if not filter_string:
        return pd.Series(True, index=df.index)
    
    # Initialize masks
    final_mask = pd.Series(True, index=df.index)
    
    # Split into AND conditions (highest precedence after parentheses and NOT)
    and_parts = split_top_level(filter_string, ';')
    
    for and_part in and_parts:
        and_part = and_part.strip()
        if not and_part:
            continue
            
        # Split into OR conditions
        or_parts = split_top_level(and_part, ',')
        or_mask = pd.Series(False, index=df.index)
        
        for or_part in or_parts:
            or_part = or_part.strip()
            if not or_part:
                continue
                
            # Process this part
            part_mask = process_part(or_part, df, depth+1)
            or_mask |= part_mask
            
        # Add to final mask with AND
        final_mask &= or_mask
    
    return final_mask


def process_part(part, df, depth):
    """Process a single part of the filter (not containing top-level AND/OR)."""
    part = part.strip()
    
    # Check for NOT
    if part.startswith('~'):
        # Process the rest and negate it
        if part[1:].strip().startswith('('):
            # Find matching closing parenthesis
            open_count = 0
            close_pos = -1
            
            for i, char in enumerate(part[1:]):
                if char == '(':
                    open_count += 1
                elif char == ')':
                    open_count -= 1
                    if open_count == 0:
                        close_pos = i + 1
                        break
            
            if close_pos > 0:
                # Extract and process the inner condition
                inner_condition = part[2:close_pos]
                inner_mask = parse_filter(inner_condition, df, depth+1)
                
                # Check if there's anything after the closing parenthesis
                if len(part) > close_pos + 1:
                    remaining = part[close_pos+1:].strip()
                    if remaining.startswith(';'):
                        # AND operation after negation
                        remaining_mask = parse_filter(remaining[1:], df, depth+1)
                        return ~inner_mask & remaining_mask
                    elif remaining.startswith(','):
                        # OR operation after negation
                        remaining_mask = parse_filter(remaining[1:], df, depth+1)
                        return ~inner_mask | remaining_mask
                
                return ~inner_mask
            else:
                raise ValueError(f"Unmatched parenthesis in: {part}")
        else:
            # Simple negation of a single condition
            return ~process_simple_condition(part[1:].strip(), df)
    
    # Check for parenthesized expression
    if part.startswith('('):
        # Find matching closing parenthesis
        open_count = 1
        close_pos = -1
        
        for i, char in enumerate(part[1:], 1):
            if char == '(':
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count == 0:
                    close_pos = i
                    break
        
        if close_pos > 0:
            # Process the inner condition
            inner_condition = part[1:close_pos]
            inner_mask = parse_filter(inner_condition, df, depth+1)
            
            # Check if there's anything after the closing parenthesis
            if len(part) > close_pos + 1:
                remaining = part[close_pos+1:].strip()
                if remaining.startswith(';'):
                    # AND operation after parenthesis
                    remaining_mask = parse_filter(remaining[1:], df, depth+1)
                    return inner_mask & remaining_mask
                elif remaining.startswith(','):
                    # OR operation after parenthesis
                    remaining_mask = parse_filter(remaining[1:], df, depth+1)
                    return inner_mask | remaining_mask
            
            return inner_mask
        else:
            raise ValueError(f"Unmatched parenthesis in: {part}")
    
    # Process simple condition
    return process_simple_condition(part, df)


def process_simple_condition(condition, df):
    """Process a simple condition without complex logical operators."""
    condition = condition.strip()
    
    # Check for double-sided range conditions like "20 < Age < 60"
    range_match = re.match(r'^(\d+(?:\.\d+)?)\s*<\s*(\w+)\s*<\s*(\d+(?:\.\d+)?)$', condition)
    if range_match:
        lower_bound, column, upper_bound = range_match.groups()
        try:
            lower_bound, upper_bound = float(lower_bound), float(upper_bound)
            return (df[column] > lower_bound) & (df[column] < upper_bound)
        except Exception as e:
            print(f"Error processing range condition: {condition} - {e}")
            return pd.Series(False, index=df.index)
    
    # Check for operators
    for operator in ['==', '!=', '<=', '>=', '<', '>']:
        if operator in condition:
            parts = condition.split(operator, 1)
            if len(parts) == 2:
                column = parts[0].strip()
                value = parts[1].strip()
                
                # Make sure the column exists
                if column not in df.columns:
                    print(f"Warning: Column '{column}' not found in DataFrame")
                    return pd.Series(False, index=df.index)
                
                # Try to convert to appropriate type
                try:
                    val = int(value)
                except ValueError:
                    try:
                        val = float(value)
                    except ValueError:
                        val = value
                
                # Apply the appropriate comparison
                try:
                    if operator == '==':
                        return df[column] == val
                    elif operator == '!=':
                        return df[column] != val
                    elif operator == '<':
                        return df[column] < val
                    elif operator == '>':
                        return df[column] > val
                    elif operator == '<=':
                        return df[column] <= val
                    elif operator == '>=':
                        return df[column] >= val
                except Exception as e:
                    print(f"Error comparing {column} {operator} {value}: {e}")
                    return pd.Series(False, index=df.index)
    
    # If we get here, we couldn't parse the condition
    print(f"Warning: Could not parse condition: {condition} - treating as False")
    return pd.Series(False, index=df.index)


def split_top_level(text, delimiter):
    """
    Split text by delimiter, but only at the top level (not inside parentheses).
    If delimiter is ',' and text contains '==', prefix each part after splitting with the pattern before '=='.
    """
    parts = []
    current_part = ""
    paren_level = 0
    
    for char in text:
        if char == '(':
            paren_level += 1
            current_part += char
        elif char == ')':
            paren_level -= 1
            current_part += char
        elif char == delimiter and paren_level == 0:
            parts.append(current_part)
            current_part = ""
        else:
            current_part += char
    
    if current_part:
        parts.append(current_part)
    
    # Apply the prefix logic only if delimiter is ',' and the pattern has '=='
    if delimiter == ',' and parts and '==' in parts[0]:
        # Extract the prefix from the first part
        prefix_parts = parts[0].split('==')
        prefix = prefix_parts[0] + '=='
        
        # Keep first part as is, modify the rest
        final_parts = [parts[0]]
        for part in parts[1:]:
            final_parts.append(prefix + part.strip())
        
        return final_parts
    
    return parts

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


# Function to calculate effect size (Conhen's d for case-control contrast) and CI
def CohenD_CI(t_matrix, n1_matrix, n2_matrix, alpha=0.05, output_path=None):
    """
    Calculate Cohen's d, confidence intervals, and related statistics voxel-wise.
    
    Parameters:
    -----------
    t_matrix : numpy.ndarray or str
        Input t-value matrix or path to NIfTI/CSV file containing t-values
    n1_matrix : numpy.ndarray or str
        Input matrix or file for sample sizes of group 1
    n2_matrix : numpy.ndarray or str
        Input matrix or file for sample sizes of group 2
    alpha : float, optional
        Significance level for confidence interval (default: 0.05)
    output_path : str, optional
        Directory path to save output files
    
    Returns:
    --------
    Comprehensive dictionary of statistical results
    """
    # Determine input format
    def load_input(input_data):
        if isinstance(input_data, str):
            if input_data.endswith('.nii') or input_data.endswith('.nii.gz'):
                # Load NIfTI image
                nifti_img = nib.load(input_data)
                return nifti_img.get_fdata(), True, nifti_img
            elif input_data.endswith('.csv'):
                # Load CSV
                return pd.read_csv(input_data, header=None).to_numpy(), False, None
            else:
                raise ValueError("Unsupported file format. Use .nii, .nii.gz, or .csv")
        else:
            # Assume numpy array
            return np.asarray(input_data), False, None
    
    # Load inputs
    t_values, is_nifti, nifti_img = load_input(t_matrix)
    n1_values, _, _ = load_input(n1_matrix)
    n2_values, _, _ = load_input(n2_matrix)
    
    # Validate input dimensions
    if not (t_values.shape == n1_values.shape == n2_values.shape):
        raise ValueError("Input matrices must have identical dimensions")
    
    # Degrees of freedom
    df = n1_values + n2_values - 2
    
    # Cohen's d calculation (voxel-wise)
    # d = t * sqrt((n1 + n2) / (n1 * n2))
    d_values = t_values * np.sqrt(1/n1_values + 1/n2_values)
    
    # Standard Error calculation
    se_diff = np.sqrt(1/n1_values + 1/n2_values)
    
    # Calculate critical value
    critical_value = stats.t.ppf(1 - alpha / 2, df)
    
    # Confidence Interval
    ci_lower = d_values - (critical_value * se_diff)
    ci_upper = d_values + (critical_value * se_diff)
    
    # Effect direction (based on original t-values)
    effect_direction = np.sign(t_values)
    
    # Interpretation of effect size
    def interpret_effect_size(d):
        abs_d = np.abs(d)
        interpretation = np.full_like(abs_d, fill_value='Negligible effect', dtype='<U20')
        interpretation[abs_d >= 0.2] = 'Small effect'
        interpretation[abs_d >= 0.5] = 'Medium effect'
        interpretation[abs_d >= 0.8] = 'Large effect'
        return interpretation
    
    # Prepare results
    results = {
        'cohens_d': d_values,  # Preserves original sign from t-values
        'cohens_d_abs': np.abs(d_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'effect_direction': effect_direction,
        'interpretation': interpret_effect_size(np.abs(d_values)),
        'sample_info': {
            'sample_size_1': n1_values,
            'sample_size_2': n2_values,
            'degrees_of_freedom': df
        }
    }
    
    # Handle output
    if output_path:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare output files
        output_files = {
            'cohens_d': os.path.join(output_path, 'cohens_d.csv'),
            'ci_lower': os.path.join(output_path, 'ci_lower.csv'),
            'ci_upper': os.path.join(output_path, 'ci_upper.csv')
        }
        
        # Save based on input format
        if is_nifti:
            # Create and save NIfTI images
            for key, filepath in output_files.items():
                output_nifti = nib.Nifti1Image(results[key], nifti_img.affine, nifti_img.header)
                nib.save(output_nifti, filepath.replace('.csv', '.nii.gz'))
        else:
            # Save as CSV files
            for key, filepath in output_files.items():
                pd.DataFrame(results[key]).to_csv(filepath, index=False, header=False)
        
        return results
    
    # Always return results dictionary
    return results

# Function to extract variable names from a statistical model string
def extract_model_variables(model_txt, random_factor_included=False):
    """
    Extract variable names from a statistical model string.
    
    This function parses a model string (typically from a formula in statistical modeling)
    and extracts the unique variable names, removing operators and numeric values.
    
    Args:
        model_txt (str): A statistical model string, typically in the format '~var1 + var2 + ...'
        random_factor_included (bool, optional): If False, variables after '|' are not included.
                                                 Defaults to False.
    
    Returns:
        list: A list of unique variable names found in the model string
    
    Raises:
        ValueError: If the model string does not contain '~' and ')' 
                    (typical of many statistical model representations)
    
    Example:
        >>> extract_model_variables('~var1 + var2 * (1|SITE)', random_factor_included=False)
        ['var1', 'var2']
        >>> extract_model_variables('~var1 + var2 * (1|SITE)', random_factor_included=True)
        ['var1', 'var2', 'SITE']
    """
    # Find the part of the string after '~' and before ')'
    match = re.search(r'~(.*)\)', model_txt)
    
    # Raise an error if the expected pattern is not found
    if not match:
        raise ValueError("Invalid model string format. Expected '~' and ')'")
    
    # Extract the variables portion of the string
    model_vars = match.group(1)
    
    # If random factors are not to be included, remove everything after '|'
    if not random_factor_included:
        model_vars = model_vars.split('|')[0]
    
    # Replace various statistical model separators with spaces
    # This helps in splitting the string into individual components
    for sep in ['+', '-', '(', ')', '|', '*', ':','/']:
        model_vars = model_vars.replace(sep, ' ')
    
    # Split by whitespace and clean up the variable list
    # Remove empty strings and strip whitespace
    var_list = [var.strip() for var in model_vars.split() if var.strip()]
    
    # Remove any numeric values that might have been extracted
    var_list = [var for var in var_list if not var.isdigit()]
    
    # Return unique variables while preserving order
    return list(dict.fromkeys(var_list))

# Function to find categorical columns in a CSV file
def find_categorical_columns(csv_path, columns_to_check=None):
    """
    Find categorical columns in a CSV file.
    
    A column is considered categorical if it has:
    - More than 1 and fewer than 5 unique non-missing values
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file to be loaded
    
    columns_to_check : list, optional
        List of column names to check. If None, checks all columns.
    
    Returns:
    --------
    list or None
        A list of column names that are categorical.
        Returns None if no categorical columns are found.
    
    Example:
    --------
    >>> find_categorical_columns('data.csv', columns_to_check=['Age', 'Sex'])
    ['Sex']
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # If no columns specified, check all columns
    if columns_to_check is None:
        columns_to_check = df.columns
    
    # List to store categorical column names
    categorical_columns = []
    
    # Check each specified column
    for col in columns_to_check:
        # Check if column exists
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")
        
        # Count unique non-missing values
        unique_non_missing_values = df[col].dropna().nunique()
        
        # Check if column is categorical
        if 1 < unique_non_missing_values < 5:
            categorical_columns.append(col)
    
    # Return None if no categorical columns found
    return categorical_columns if categorical_columns else None

# Function to make descriptive statistics Table 1 & Table S1
def generate_demographic_tables(
    model_name: str,
    model_txt: str, 
    subjects_csv_file: str, 
    table1_group_var: str, 
    table1_site_var: str = None
) -> tuple:
    """
    Generate demographic tables from a statistical model and CSV file.

    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "M01" or "Model_02") to be used in output file names
    model_txt : str
        Statistical model formula (e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1/SITE|site))")
    subjects_csv_file : str
        Path to the CSV file containing subject information
    table1_group_var : str
        Variable to use for primary grouping in Table 1
    table1_site_var : str, optional
        Variable to use for secondary grouping in Table S1

    Returns:
    --------
    tuple
        A tuple containing:
        - DataFrame of Table 1 
        - DataFrame of Supplementary Table 1 (if site_var is provided)
    """
    # Determine output directory (same as the directory of the CSV file)
    output_dir = os.path.dirname(os.path.abspath(subjects_csv_file))
    
    # Step 1: Extract variables from the model string
    match = re.search(r'~(.*)\)', model_txt)
    if not match:
        raise ValueError("Invalid model string format. Expected '~' and ')'")
    
    model_vars = match.group(1)
    
    # Replace separators with spaces
    for sep in ['+', '-', '(', ')', '|', '*', ':', '/']:
        model_vars = model_vars.replace(sep, ' ')
    
    # Split and clean variable list
    var_list = [var.strip() for var in model_vars.split() if var.strip()]
    var_list = [var for var in var_list if not var.isdigit()]
    
    # Load CSV file
    print(f"Loading subjects' information from {subjects_csv_file}...")
    df0 = pd.read_csv(subjects_csv_file)
    
    # Create DataFrame with specified columns, dropping NA
    df = df0[var_list].copy().dropna()
    
    # Generate primary Table 1 (by primary group variable)
    # Exclude site variable if provided
    df1 = df.drop(table1_site_var, axis=1) if table1_site_var else df
    
    # # For fine tune purpose only
    # df2 = df1.copy() # keep an independent copy 
    # # Remove rows where GROUP is 0 AND AGE is less than specified age
    # # df1 = df1[~(df1["AGE"] < 18.1)]
    # df1 = df1[~((df1["GROUP"] == 0) & (df1["AGE"] < 13.1))]
    # # df1 = df2.copy() # put the original dataframe back
    
    table1 = TableOne(
        df1, 
        groupby=table1_group_var, 
        decimals=3,
        pval=True, 
        htest_name=True,
        dip_test=True, 
        normal_test=True, 
        tukey_test=True
    )
    print(table1)
    
    # Save primary table with model name
    table1_filename = f"{model_name}_Table_1.csv"
    table1_path = os.path.join(output_dir, table1_filename)
    table1.to_csv(table1_path)
    print(f"Primary table saved to {table1_path}")
    
    # Generate supplementary table by site (if site variable provided)
    tables1 = None
    if table1_site_var:
        tables1 = TableOne(
            df, 
            groupby=table1_site_var, 
            decimals=3
        )
        
        # Save supplementary table with model name
        tables1_filename = f"{model_name}_Table_S1.csv"
        tables1_path = os.path.join(output_dir, tables1_filename)
        tables1.to_csv(tables1_path)
        print(f"Supplementary table saved to {tables1_path}")
    
    return table1, tables1

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
        
    def filter(self, all_Subjects, filter_string, model_Subjects, model_name, model_formula, table1_site_var=None, table1_group_var=None):
        """
        Filter a pandas DataFrame using a text-based filter condition
        
        Args:
            all_Subjects (str): Path to the 'Subjects.csv' in the Processes folder.
            filter_string (str): Texts to filter rows of interest, very flexible, such as
                - "20 < Age < 60; Sex == Female; Gender == 1; Site == Duke, Emory; Sev == 0,1"
                - "20 < Age < 30, Age > 60; Sex == Female"
            model_Subjects (str): Path to the 'Subjects.csv' in Results/model-specific folder.
            model_name (str): Text of statistical model name.
            model_formula (str): Test of statistical model formula, e.g. "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))"
            table1_site_var (str): Optional, for site-specific descriptive statistics Table S1.
            table1_group_var (str): Optional, for descriptive statistics Table 1 & Table S1.
        
        Outputs:
            save filtered dataframe into model-specific .csv.
        """     

        t0 = time.time()  # Record the start time
        
        # Make the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(model_Subjects), exist_ok=True) 
        
        # Read Subjects.csv for all datatype and models
        na_values = ["NA","na","N/A","n/a","None","none","NONE","NaN","nan","Nan","NAN","","NULL","null","Null"]
        df = pd.read_csv(all_Subjects, na_values = na_values, keep_default_na=True)
        
        # Filter if the filter string exists
        df = filter_dataframe(df, filter_string) if filter_string is not None else df
        
        # Save to model-specific csv
        df.to_csv(model_Subjects, index=False)
        print(f"Model-specific csv file made: {model_Subjects}")
        
        # Make descriptive reports Table 1 & Table S1
        try:
            table1, tables1 = generate_demographic_tables(model_name, model_formula, model_Subjects, table1_group_var, table1_site_var)
        except Exception as e:
            print(f"An error occurred for making Table 1 or Table S1: {e}")
        
        # Print ending info
        print(f"Model-specific filtering completed!\nTime elapsed (in secs): {time.time()-t0}\n")
              
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

    def reverse(self, process_dir, result_dir, model_name, model_formula, model_Subjects, mask1, path_R_pTFCE, my_rois_path):
        """
        (1) Apply inclusive mask (if available) to restrict all statistical outputs within the mask
        (2) FDR correction (default) for p-values; 
        (3) Negatively log10 transformed p-values; 
        (4) Effect size;
        (5) Reverse the concatenate CSV files of statistical outputs back to its original dimensions;
        (6) pTFCE for .nii and .nii.gz;
        (7) Html report.
        
        Args:
            process_dir (str):  Path to the folder of the target data type in the Process folder.
            result_dir (str):   Path to the folder of statistical outputs of the target data type.
            model_name (str):   Name of the model, e.g., 'model_01'.
            model_formula(str): Texts of the model, e.g., 'lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))'.
            model_Subjects(str):Path to the datatype- and model-specific Subjects's .csv file.
            mask1 (str):        Path to the inclusive mask image, e.g., '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/brain_mask.nii'.
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
        if mask1 is None or mask1 == "None" or (not isinstance(mask1, str) and pd.isna(mask1)):
            pass
        else: 
            # flatten the mask1 & save into mask1.csv
            df_mask = flatten_single((mask1,result_dir,'mask1',None))
            # apply mask to all csv files & update the csv files
            # apply_mask_to_csv_files(df_mask, result_dir)
                # List all CSV files recursively
            csv_files = [os.path.join(root, file) 
                        for base_dir in [os.path.join(result_dir, "Mega"), os.path.join(result_dir, "Meta")]
                        for root, _, files in os.walk(base_dir) if os.path.exists(base_dir)
                        for file in files if file.endswith('.csv')]
            
            # Prepare arguments for multiprocessing
            args = [(csv_file, df_mask['Yvar'].tolist()) for csv_file in csv_files]
            
            # # For test purpose only
            # apply_mask_to_single_csv_file(args[0])
            
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
        
        # Define the specific directories to search within
        search_dirs = [os.path.join(result_dir, 'Mega'), os.path.join(result_dir, 'Meta')]

        # List comprehension version with recursive search in specific dirs
        csv_files = [
            os.path.join(root, file)
            for base_dir in search_dirs
            if os.path.exists(base_dir)
            for root, dirs, files in os.walk(base_dir)
            for file in files
            if (file == (model_name + ".csv") or file == ("neg_log10_" + model_name + ".csv"))
        ]
        
        # Length of flattened but not segmented data
        numbers = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(os.path.join(process_dir, 'segmented')) if f.startswith('V') and f.endswith('.csv') and '_' in f and f.split('_')[1].split('.')[0].isdigit()]
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
            if f_data[0].endswith('.tsv'):
                data = np.genfromtxt(f_data[0], delimiter='\t')
            else:  # Default to CSV for other extensions
                data = np.genfromtxt(f_data[0], delimiter=',')
            # Is Symmetric
            is_symmetric = np.allclose(data, data.T, equal_nan=True) if data.ndim == 2 and data.shape[0] == data.shape[1] else None
            file_type = 'Symmetric Matrix' if is_symmetric else 'CSV' # file type for .csv files
            sample_file = None

        # Arguments
        my_args = [(csv_file, total_length, file_type, sample_file)
                   for csv_file in csv_files]
        # # Print
        # for t in my_args:
        #     print(t)
        
        # # For test purpose only !!
        # reverse_single(my_args[10])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(reverse_single, my_args) 
            
        # Print ending info
        print(f"Reverse statistical outputs back to original dimensions completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
        
        ## (5) pTFCE
        if file_extension in ['.nii', '.nii.gz']:
            # Print information
            t0 = time.time() # start time
            print(f"\npTFCE for .nii and .nii.gz ... ")
            
            # Folder of statistic
            folder_statistic = os.path.join(result_dir,'Mega','TIDY','statistic')
            
            # List all .nii and .nii.gz recursively in TIDY/statistic folder of the given model
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
            
            # # For test purpose only !!
            r_script2(pool_args[0])

            # Parallel processing
            with mp.Pool(processes=self.num_processes) as pool:
                pool.map(r_script2, pool_args)
            
            # Print ending info
            print(f"pTFCE completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        
        ## (6) Effect Size (categorical variables only)
        # Print information
        t0 = time.time() # start time
        print(f"\nEffect size calculation ... ")
        
        # Extract variables
        model_variables = extract_model_variables(model_formula)
        
        # Find categorical variables
        list_var = find_categorical_columns(model_Subjects, columns_to_check=model_variables)
        
        # Run
        if list_var is not None:
            for var in list_var:
                Z0 = os.path.join(result_dir,"Mega","TIDY","statistic",var,"OUT_" + model_name + ".nii.gz_pTFCE","Zmap.nii.gz")
                if os.path.isfile(Z0): # for NIFTI
                    N1 = os.path.join(result_dir,"Mega","GLANCE","nobs_" + var + "_0","OUT_" + model_name + ".nii.gz")
                    N2 = os.path.join(result_dir,"Mega","GLANCE","nobs_" + var + "_1","OUT_" + model_name + ".nii.gz")
                else: # for csv files
                    Z0 = os.path.join(result_dir,"Mega","TIDY","statistic",var,"OUT_" + model_name + ".csv")
                    N1 = os.path.join(result_dir,"Mega","GLANCE","nobs_" + var + "_0","OUT_" + model_name + ".csv")
                    N2 = os.path.join(result_dir,"Mega","GLANCE","nobs_" + var + "_1","OUT_" + model_name + ".csv")
                
                Out0 = os.path.join(result_dir,"Mega","TIDY","effect_size",var,"OUT_" + model_name)
                CohenD_CI(Z0, N1, N2, alpha=0.05, output_path=Out0)
                # Print ending info
                print(f"Effect size calculation completed!\nTime elapsed (in secs): {time.time()-t0}\n")
        else:
            print("No mean effects of categorical variables") 
        
        # # Make results report (one .xlsx file per model, one sheet per effect)
        # # Check if the data in the masked folder is 1D (i.e., ROI-based data)
        # data = np.genfromtxt(f_data[0], delimiter=',') # load the 1st .csv file
        # if data.ndim == 1:
        #     my_rois = pd.read_excel(my_rois_path, sheet_name='MY_ROIs', dtype='object')
        #     roi_results(model_name, csv_files, my_rois)
        
        # # Print ending info
        # print(f"ROI results report completed!\nTime elapsed (in secs): {time.time()-t0}\n")
 