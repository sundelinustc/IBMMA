import os
import multiprocessing as mp # for parallel processing
import shutil
import numpy as np
import pandas as pd
import re
from tableone import TableOne, load_dataset # for Table1
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from nilearn.plotting import find_probabilistic_atlas_cut_coords, find_parcellation_cut_coords
from nilearn.reporting import get_clusters_table
from nilearn.image import threshold_img, math_img, load_img
from nilearn.datasets import load_mni152_template
import seaborn as sns
from datetime import datetime
from atlasreader import create_output  # tables of brain regions & brain maps
import time # for measuring time elapsed


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
    for sep in ['+', '-', '(', ')', '|', '*', ':']:
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

# Function to report statistical results of voxel-wised NIFTI
def generate_nifti_report(
    subjects_csv_file,
    stat_img_path=None,
    pos_map_path=None,
    neg_map_path=None,
    thres_txt_path=None,
    output_dir=None,
    data_type=None,
    model_num=None,
    model_txt=None,
    effect_interest=None,
    cluster_extent=20,
    atlas=None,
    direction='both',
    prob_thresh=0,
    min_distance=8,
    peak_region_min_percentage=10,
    cluster_region_min_volume=10,
    overwrite=True,
    table1_site_var=None, 
    table1_group_var=None
):
    """
    Generate a comprehensive neuroimaging report from statistical maps.
    
    Parameters:
    -----------
    subjects_csv_file : str
        Path to the Subjects.csv.
    stat_img_path : str
        Path to the z-score map (both positive and negative, NIFTI file).
    pos_map_path : str
        Path to the positive z-score map (NIFTI file).
    neg_map_path : str
        Path to the negative z-score map (NIFTI file, values are positive).
    thres_txt_path : str
        Path to text file containing thresholds for p<sub>FWE</sub> &lt; 0.05.
    output_dir : str
        Directory where the report will be saved.
    data_type : str
        Type of data being analyzed.
    model_num : str
        Model number or identifier.
    model_txt : str
        Text description of the statistical model used.
    effect_interest : str
        Effect of interest in the analysis.
    cluster_extent : int, optional
        Minimum number of voxels within a cluster (default: 20).
    atlas : list or None, optional
        List of atlases to use for labeling (default: a comprehensive list of available atlases).
    direction : str, optional
        Direction of effects to include: 'pos', 'neg', or 'both' (default: 'both').
    prob_thresh : int, optional
        Probability threshold to apply to atlas (default: 0).
    min_distance : int, optional
        Minimum distance (in mm) between sub-peaks in a cluster (default: 8).
    peak_region_min_percentage : float, optional
        Minimum percentage threshold for displaying peak regions (default: 10).
    cluster_region_min_volume : float, optional
        Minimum volume threshold (percentage * volume) for displaying cluster regions (default: 10 mm3).
    overwrite : bool, optional
        Whether to overwrite existing output directory (default: True).
    
    Returns:
    --------
    str
        Path to the generated HTML report.
    """
    # Set default atlas list if none provided
    if atlas is None:
        atlas = [
            "harvard_oxford", "aal", "desikan_killiany", "destrieux",
            "marsatlas", "aicha", "juelich", "neuromorphometrics",
            "talairach_ba", "talairach_gyrus"
        ]
    
    # Create report directory
    report_dir = output_dir
    html_dir = report_dir
    stat_img_path = os.path.join(report_dir, "stat_img.nii.gz")
    
    # Create output directory
    if overwrite and os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    # Step 0: Make Table 1 for Demographic and Clinical Information
    # Extract variable names from the given statistical model string
    match = re.search(r'~(.*)\)', model_txt)
    if not match:
        raise ValueError("Invalid model string format. Expected '~' and ')'")
    
    model_vars = match.group(1)
    
    # Split by separators: +, -, (, |, *, :
    # First replace these with a common separator
    for sep in ['+', '-', '(', ')','|', '*', ':', '/']:
        model_vars = model_vars.replace(sep, ' ')
    
    # Split by whitespace and remove empty strings
    var_list = [var.strip() for var in model_vars.split() if var.strip()]
    
    # Remove numeric values (like "1" in "(1|SITE)")
    var_list = [var for var in var_list if not var.isdigit()]
    
    # Load the CSV file as a DataFrame
    print(f"Loading subjects' infomation from {subjects_csv_file}...")
    df0 = pd.read_csv(subjects_csv_file)
    
    # Create a new DataFrame with only the specified columns
    df = df0[var_list].copy().dropna()
    
    # Make Table1 (by GROUP)
    df1= df.drop(table1_site_var,axis=1)
    table1 = TableOne(df1, groupby=table1_group_var, decimals=3, 
                    pval = True, htest_name=True,
                    dip_test=True, normal_test=True, tukey_test=True)
    table1.to_csv(os.path.join(output_dir,'table1.csv'))
    
    # Make TableS1 (by GROUP & SITE)
    tables1 = TableOne(df, groupby=table1_site_var, decimals=3)
    tables1.to_csv(os.path.join(output_dir,'tables1.csv'))
    
    # Step 1: Load the txt file and extract thresholds
    with open(thres_txt_path, 'r') as f:
        lines = f.readlines()
    
    threshold1 = float(lines[0][:3]) # the numeric value at the beginning of the 1st line
    threshold2 = float(lines[1][:3]) # the numeric value at the beginning of the 2nd line
    voxel_thresh = min(threshold1, threshold2)  # make sure that both pos & neg are significant
    
    # Step 2: Load and threshold the positive image
    img1 = nib.load(pos_map_path)
    data1 = img1.get_fdata()
    thresholded_data1 = np.where(data1 > 0, data1, 0)  # Z > 0, keep all positive values
    
    # Step 3: Load and threshold the negative image
    img2 = nib.load(neg_map_path)
    data2 = img2.get_fdata()
    thresholded_data2 = np.where(data2 > 0, -1 * data2, 0)  # Negate values to represent negative effects
    
    # Step 4: Merge the two thresholded images & save it
    merged_data = thresholded_data1 + thresholded_data2
    merged_img = nib.Nifti1Image(merged_data, img1.affine, img1.header)
    nib.save(merged_img, stat_img_path)
    
    # Step 5: Create output using atlasreader
    create_output(
        filename=stat_img_path,
        cluster_extent=cluster_extent,
        atlas=atlas,
        voxel_thresh=voxel_thresh,
        direction=direction,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        glass_plot_kws={"black_bg": False, "cmap": "RdBu_r"},
        stat_plot_kws={"black_bg": False, "title": None, "cmap": "RdBu_r"},
        outdir=report_dir
    )
    
    # Step 6: Load peak and cluster data
    peaks_file = os.path.join(report_dir, "stat_img_peaks.csv")
    clusters_file = os.path.join(report_dir, "stat_img_clusters.csv")
    
    # Step 7: Generate brain visualizations
    # Surface view (interactive)
    stat_img = nib.load(stat_img_path)
    surface_view = plotting.view_img_on_surf(
        # title=f'Surface Projection',
        stat_img,
        threshold=voxel_thresh,
        surf_mesh='fsaverage',
        cmap="RdBu_r",
        black_bg=False
    )
    surface_view.save_as_html(os.path.join(html_dir, "surface_view.html"))
    
    # Slice views with anatomical underlay
    template = load_mni152_template()
    
    # Axial slices
    plotting.plot_stat_map(
        stat_img,
        bg_img=template,
        threshold=voxel_thresh,
        cut_coords=10,
        display_mode='z',
        cmap="RdBu_r",
        black_bg=False,
        output_file=os.path.join(report_dir, "slice_view_axial.png"),
        title=None
    )
    
    # Sagittal slices
    plotting.plot_stat_map(
        stat_img,
        bg_img=template,
        threshold=voxel_thresh,
        cut_coords=10,
        display_mode='x',
        cmap="RdBu_r",
        black_bg=False,
        output_file=os.path.join(report_dir, "slice_view_sagittal.png"),
        title=None
    )
    
    # Coronal slices
    plotting.plot_stat_map(
        stat_img,
        bg_img=template,
        threshold=voxel_thresh,
        cut_coords=10,
        display_mode='y',
        cmap="RdBu_r",
        black_bg=False,
        output_file=os.path.join(report_dir, "slice_view_coronal.png"),
        title=None
    )
    
    # Interactive statistical map view
    interactive_view = plotting.view_img(
        stat_img,
        threshold=voxel_thresh,
        cmap="RdBu_r",
        black_bg=False,
        colorbar=True
    )
    interactive_view.save_as_html(os.path.join(html_dir, "interactive_map.html"))
    
    # Load table1 and tables1 for HTML display
    table1_html = ""
    tables1_html = ""
    
    # Load and format table1.csv to HTML
    table1_path = os.path.join(output_dir, 'table1.csv')
    if os.path.exists(table1_path):
        table1_df = pd.read_csv(table1_path, skiprows=1)
        # Replace NaN with empty string
        table1_df = table1_df.fillna("")
        # Rename columns
        if "Unnamed: 0" in table1_df.columns:
            table1_df = table1_df.rename(columns={"Unnamed: 0": "Variables"})
        if "Unnamed: 1" in table1_df.columns:
            table1_df = table1_df.rename(columns={"Unnamed: 1": ""})
        table1_html = table1_df.to_html(index=False, classes='table table-striped', border=0)
    
    # Load and format tables1.csv to HTML (skipping first row and transposing)
    tables1_path = os.path.join(output_dir, 'tables1.csv')
    if os.path.exists(tables1_path):
        # Skip the first row and transpose
        tables1_df = pd.read_csv(tables1_path, skiprows=1)
        # Replace NaN with empty string
        tables1_df = tables1_df.fillna("")
        # Rename columns
        if "Unnamed: 0" in tables1_df.columns:
            tables1_df = tables1_df.rename(columns={"Unnamed: 0": "SITE"})
        if "Unnamed: 1" in tables1_df.columns:
            tables1_df = tables1_df.rename(columns={"Unnamed: 1": ""})
        # Then use the first row as headers and drop that row
        tables1_df = tables1_df.transpose()
        tables1_df.columns = tables1_df.iloc[0]
        tables1_df = tables1_df.drop(tables1_df.index[0]) 
        tables1_html = tables1_df.to_html(index=True, classes='table table-striped', border=0)
    
    # Step 8: Create comprehensive HTML report
    html_path = os.path.join(html_dir, "index.html")
    
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neuroimaging Analysis Results</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                .visualization {{ margin: 30px 0; background: #f9f9f9; padding: 20px; border-radius: 5px; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
                .section {{ margin-bottom: 40px; }}
                footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.8em; color: #777; }}
                .code {{ font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .info-table {{ width: 100%; margin-bottom: 20px; }}
                .info-table td {{ padding: 8px; vertical-align: top; }}
                .info-table td:first-child {{ font-weight: bold; width: 200px; }}
                .footnotes {{ margin-top: 15px; font-size: 0.9em; background-color: #f8f8f8; padding: 15px; border-radius: 5px; }}
                .footnotes ol {{ padding-left: 20px; }}
                .footnotes li {{ margin-bottom: 5px; }}
                .table-responsive {{ overflow-x: auto; }}
                .table-striped tbody tr:nth-of-type(odd) {{ background-color: rgba(0,0,0,.05); }}
            </style>
        </head>
        <body>
            <h1>Neuroimaging Statistical Analysis Results</h1>
            
            <div class="section">
                <h2>Analysis Information</h2>
                <table class="info-table">
                    {"<tr><td>Data Type</td><td>" + str(data_type) + "</td></tr>" if data_type else ""}
                    {"<tr><td>Model Number</td><td>" + str(model_num) + "</td></tr>" if model_num else ""}
                    {"<tr><td>Model Equation</td><td>" + str(model_txt) + "</td></tr>" if model_txt else ""}
                    {"<tr><td>Effect of Interest</td><td>" + str(effect_interest) + "</td></tr>" if effect_interest else ""}
                </table>
            </div>
            
            <div class="section">
                <h2>File Paths</h2>
                <table class="info-table">
                    <tr><td>Positive Map:</td><td>{pos_map_path}</td></tr>
                    <tr><td>Negative Map:</td><td>{neg_map_path}</td></tr>
                    <tr><td>Threshold File:</td><td>{thres_txt_path}</td></tr>
                    <tr><td>Output Directory:</td><td>{output_dir}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Methods</h2>
                <p>Statistical analysis was conducted using the <a href="https://github.com/sundelinustc/IBMMA" target="_blank">IBMMA</a> package for neuroimaging data processing. Statistical inference was enhanced through the implementation of the probabilistic threshold-free cluster enhancement (<a href="https://spisakt.github.io/pTFCE/" target="_blank">pTFCE) method</a>, which optimizes the detection of significant signals by combining cluster extent and voxel intensity information. Two-tailed statistical maps were thresholded at Z<sub>pTFCE</sub> &gt; {voxel_thresh} 
                (corresponding to p<sub>FWE</sub> &lt; 0.05) with a minimum cluster extent threshold of {cluster_extent} contiguous voxels to control for multiple comparisons. All visualizations presented herein display only voxels that exceeded these statistical thresholds.</p>
                <p>Anatomical localization of significant clusters and peaks was performed using the <a href="https://nilearn.github.io/stable/modules/generated/nilearn.reporting.get_clusters_table.html" target="_blank">Nilearn</a> and the <a href="https://github.com/miykael/atlasreader" target="_blank">AtlasReader</a> packages, which provide standardized labeling according to multiple neuroanatomical atlases for comprehensive spatial interpretation of results.</p>
            </div>

            <div class="section">
                <h2>Results</h2>
            </div>
            
            <div class="section">
                <h2>Table 1. Demographic and Clinical Information</h2>
                <div class="table-responsive">
                    {table1_html}
                    <p><em>Note: Group comparisons of demographic and clinical variables.</em></p>
                </div>
            </div>
            
            <div class="section">
                <h2>Table S1. Site-Specific Information</h2>
                <div class="table-responsive">
                    {tables1_html}
                    <p><em>Note: Distribution of variables across different sites.</em></p>
                </div>
            </div>
            
            <div class="section">
                <h2>Slice Views</h2>
                
                <div>
                    <h3>Axial Slices</h3>
                    <img src="slice_view_axial.png" alt="Axial Slices">
                </div>
                
                <div>
                    <h3>Sagittal Slices</h3>
                    <img src="slice_view_sagittal.png" alt="Sagittal Slices">
                </div>
                
                <div>
                    <h3>Coronal Slices</h3>
                    <img src="slice_view_coronal.png" alt="Coronal Slices">
                </div>
            </div>
            
            <div class="section">
                <h2>Interactive Visualizations</h2>
                
                <div>
                    <h3>Surface Projection</h3>
                    <iframe src="surface_view.html" style="width: 100%; height: 600px; border: none;"></iframe>
                </div>
                
                <div>
                    <h3>3D View</h3>
                    <p>Nevigate the brain maps through left-click.</p>
                    <iframe src="interactive_map.html" style="width: 100%; height: 600px; border: none;"></iframe>
                </div>
            </div>
        """)
        
        # Add cluster visualizations if they exist
        cluster_images = [f for f in os.listdir(report_dir) if f.startswith("stat_img_cluster") and f.endswith(".png")]
        if cluster_images:
            f.write("""
            <div class="section">
                <h2>Significant Clusters</h2>
                <p>The images below show the significant clusters identified in the analysis.</p>
            """)
            
            for img in sorted(cluster_images):
                f.write(f"""
                <div class="visualization">
                    <h3>{img.replace('.png', '')}</h3>
                    <img src="{img}" alt="{img.replace('.png', '')}">
                </div>
                """)
            
            f.write("</div>")
        
        # Merge peaks and clusters tables if they exist
        if os.path.exists(peaks_file) and os.path.exists(clusters_file):
            # Load the peaks data
            peaks = pd.read_csv(peaks_file)
 
        # Peak values corrected by suing Nilearn function
        res_df = get_clusters_table(
            stat_img,
            stat_threshold=voxel_thresh,
            cluster_threshold=cluster_extent,
            two_sided=True,
            min_distance=min_distance
        )
        res_df.to_csv('Nilearn_table.csv')
            
        # Convert cluster_id column to integer if it exists
        if 'cluster_id' in peaks.columns:
            peaks['cluster_id'] = peaks['cluster_id'].astype(int)
            
            # Process peak regions to filter by percentage
            def filter_peak_regions(row):
                regions = []
                if pd.notna(row['harvard_oxford']):
                    # Split by semicolon
                    parts = row['harvard_oxford'].split(';')
                    for part in parts:
                        part = part.strip()
                        try:
                            # Format is "percentage% region_name"
                            if '%' in part:
                                percentage_str = part.split('%')[0].strip()
                                region_name = part.split('%')[1].strip()
                                try:
                                    percentage = float(percentage_str)
                                    if percentage >= peak_region_min_percentage:
                                        regions.append(f"{percentage:.1f}% {region_name}")
                                except ValueError:
                                    # If percentage conversion fails, include the region as is
                                    regions.append(part.strip())
                            else:
                                # If no percentage format found, include the region as is
                                regions.append(part.strip())
                        except Exception as e:
                            # If any error occurs, include the region as is
                            regions.append(part.strip())
                return '; '.join(regions) if regions else 'No regions above threshold'
            
            # Apply the peak region filter
            peaks['filtered_peak_regions'] = peaks.apply(filter_peak_regions, axis=1)
            
            # Extract and rename columns from peaks
            selected_peaks = peaks[['cluster_id', 'peak_x', 'peak_y', 'peak_z', 'peak_value', 'volume_mm']].copy()
            selected_peaks.rename(columns={
                'peak_x': 'X',
                'peak_y': 'Y',
                'peak_z': 'Z',
                'peak_value': 'Peak_Val',
                'volume_mm': 'Vol (mm3)'
            }, inplace=True)
            
            # Format the values appropriately
            selected_peaks['Peak_Val'] = selected_peaks['Peak_Val'].round(3)
            selected_peaks['Vol (mm3)'] = selected_peaks['Vol (mm3)'].astype(int)
            
            # Add the filtered peak regions
            selected_peaks['Peak_Regions'] = peaks['filtered_peak_regions']
            
            # Load the clusters data
            clusters = pd.read_csv(clusters_file)
            
            # Convert cluster_id column to integer if it exists
            if 'cluster_id' in clusters.columns:
                clusters['cluster_id'] = clusters['cluster_id'].astype(int)
            
            # Process cluster regions to filter by volume
            def filter_cluster_regions(row):
                regions = []
                vol_mm3 = row['volume_mm']
                if pd.notna(row['harvard_oxford']):
                    # Split by semicolon
                    parts = row['harvard_oxford'].split(';')
                    for part in parts:
                        part = part.strip()
                        try:
                            # Format is "percentage% region_name"
                            if '%' in part:
                                percentage_str = part.split('%')[0].strip()
                                region_name = part.split('%')[1].strip()
                                try:
                                    percentage = float(percentage_str)
                                    region_volume = (percentage / 100) * vol_mm3
                                    if region_volume >= cluster_region_min_volume:
                                        regions.append(f"{percentage:.1f}% {region_name}")
                                except ValueError:
                                    # If percentage conversion fails, include the region as is
                                    regions.append(part.strip())
                            else:
                                # If no percentage format found, include the region as is
                                regions.append(part.strip())
                        except Exception as e:
                            # If any error occurs, include the region as is
                            regions.append(part.strip())
                return '; '.join(regions) if regions else 'No regions above threshold'
            
            # Apply the cluster region filter
            clusters['filtered_cluster_regions'] = clusters.apply(filter_cluster_regions, axis=1)
            
            # Extract and rename columns from AtlasReader's cluster table
            selected_clusters = clusters[['cluster_id', 'peak_x', 'peak_y', 'peak_z','cluster_mean', 'volume_mm','filtered_cluster_regions']].copy()
            selected_clusters.rename(columns={
                'peak_x': 'X',
                'peak_y': 'Y',
                'peak_z': 'Z',
                'cluster_mean': 'Cluster_Mean',
                'volume_mm': 'Vol (mm3)',
                'filtered_cluster_regions': 'Regions'
            }, inplace=True)
            
            # Format the Cluster_Mean to 3 decimal places
            selected_clusters['Cluster_Mean'] = selected_clusters['Cluster_Mean'].round(3)
            
            # # Merge the selected data based on cluster_id (based on 2 tables of AtlasReader), BUT the peaks look incorrect
            # merged_table = pd.merge(selected_peaks, selected_clusters, on='cluster_id', how='left')
            
            # Extract and rename columns from Nilearn's cluster table (for peak info)
            selected_peaks1 = res_df[['X', 'Y', 'Z','Peak Stat']].copy()
            selected_peaks1.rename(columns={
                'Peak Stat': 'Peak_Val'
            }, inplace=True)
            # Format the Peak_Val to 3 decimal places
            selected_peaks1['Peak_Val'] = selected_peaks1['Peak_Val'].round(3)
            
            # Merge the selected data based on X/Y/Z coordinates (based on cluster table of AtlasReader and peak table of NiLearn)
            merged_table = pd.merge(selected_peaks1, selected_clusters, on=['X','Y','Z'], how='right')
            # Move column_name to first position in one line
            merged_table = merged_table[['cluster_id'] + [col for col in merged_table.columns if col != 'cluster_id']]
            
            # Save the merged table
            merged_table_path = os.path.join(report_dir, "stat_img_peaks&clusters.csv")
            merged_table.to_csv(merged_table_path, index=False)
            
            # Add the merged table to the HTML
            f.write(f"""
            <div class="section">
                <h2>Significant Peaks and Clusters</h2>
                <p>The table below shows the significant peaks and their associated clusters identified in the analysis.</p>
                <div class="table-responsive">
                    <table class='table'>
                        {merged_table.to_html(index=False, classes='table', escape=False).replace('<table class="table">', '').replace('</table>', '')}
                    </table>
                </div>
                <div class="footnotes">
                    <p><strong>Table Footnotes:</strong></p>
                    <ol>
                        <li><strong>X, Y, Z</strong>: MNI coordinates of peak locations.</li>
                        <li><strong>Peak_Val</strong>: Statistical Z<sub>pTFCE</sub> value at the peak.</li>
                        <li><strong>Vol (mm³)</strong>: Cluster volume in cubic millimeters.</li>
                        <li><strong>Peak_Regions</strong>: Harvard-Oxford Atlas regions at the peak location. Only regions with ≥{peak_region_min_percentage}% probability are displayed.</li>
                        <li><strong>Cluster_Mean</strong>: Mean statistical value within the entire cluster.</li>
                        <li><strong>Cluster_Regions</strong>: Harvard-Oxford Atlas regions within the cluster. Only regions with ≥{cluster_region_min_volume} mm³ effective volume (percentage × cluster volume) are displayed.</li>
                        <li>More detailed peak- and cluster-level results based on Harvard-Oxford, AAL, Desikan-Killiany, Destrieux, Marsatlas, AICHA, Juelich, Neuromorphometrics, Talairach-BA, and Talairach-Gyrus atlases can be found in the CSV files (stat_img_peaks.csv and stat_img_clusters.csv).</li>
                    </ol>
                </div>
            </div>""")
            
        # If only one of the tables exists, display it with original format
        elif os.path.exists(peaks_file):
            peaks = pd.read_csv(peaks_file)
            if 'cluster_id' in peaks.columns:
                peaks['cluster_id'] = peaks['cluster_id'].astype(int)
            f.write("""
            <div class="section">
                <h2>Significant Peaks</h2>
                <p>The table below shows all significant peaks identified in the analysis.</p>
                <div class="table-responsive">
            """)
            f.write(peaks.to_html(index=False, classes='table'))
            f.write("</div></div>")
            
        elif os.path.exists(clusters_file):
            clusters = pd.read_csv(clusters_file)
            if 'cluster_id' in clusters.columns:
                clusters['cluster_id'] = clusters['cluster_id'].astype(int)
            f.write("""
            <div class="section">
                <h2>Significant Clusters</h2>
                <p>The table below shows all significant clusters identified in the analysis.</p>
                <div class="table-responsive">
            """)
            f.write(clusters.to_html(index=False, classes='table'))
            f.write("</div></div>")
            
        # Add Glass Brain Visualization (at the very end)
        f.write("""
        <div class="section">
            <h2>Glass Brain Visualization</h2>
            <div>
                <h3>Results</h3>
                <img src="stat_img.png" alt="Glass Brain - All Results">
            </div>
        </div>
        """)
        
        # Add the closing footer and HTML tags
        f.write(f"""
            <footer>
                <div class="section">
                    <h2>References</h2>
                    <ol>
                        <li>Spisák, T., Spisák, Z., Zunhammer, M., Bingel, U., Smith, S., Nichols, T., & Kincses, T. (2019). Probabilistic TFCE: a generalised combination of cluster size and voxel intensity to increase statistical power. <em>Neuroimage, 185</em>, 12-26.</li>
                        <li>Notter M. P., Gale D., Herholz P., Markello R. D., Notter-Bielser M.-L., & Whitaker K. (2019). AtlasReader: A Python package to generate coordinate tables, region labels, and informative figures from statistical MRI images. <em>Journal of Open Source Software, 4(34), 1257</em>, <a href="https://doi.org/10.21105/joss.01257" target="_blank">https://doi.org/10.21105/joss.01257</a>.</li>
                    </ol>
                </div>
                
                <p style="text-align: center;">Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """)
    
    return html_path


# Function to report statistical results of atlas-based ROI-to-ROI connectome
def generate_connectome_report(subjects_csv_file, stat_file, pval_file, labels_file, atlas_file=None, pval_threshold=0.05, output_dir="output", data_type=None, model_num=None, model_txt=None, effect_interest=None, table1_site_var=None, table1_group_var=None):
    """
    Process correlation matrices with statistical thresholding and visualization.
    
    Parameters:
    -----------
    subjects_csv_file : str
        Path to the Subjects.csv.
    stat_file : str
        Path to the CSV file containing the statistics matrix.
    pval_file : str
        Path to the CSV file containing the p-values matrix.
    labels_file : str
        Path to the text file containing node labels (id and name).
    atlas_file : str, optional
        Path to the atlas NIfTI file for extracting node coordinates.
    pval_threshold : float
        P-value threshold for statistical significance.
    output_dir : str
        Directory to save output files.
    data_type : str
        Type of data being analyzed.
    model_num : str
        Model number or identifier.
    model_txt : str
        Text description of the statistical model used.
    effect_interest : str
        Effect of interest in the analysis.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Extract atlas names from data_type for methods section
    atlas_names = []
    if data_type:
        # Find all atlas-X patterns in data_type
        atlas_matches = re.finditer(r'atlas-(\w+)', data_type)
        for match in atlas_matches:
            atlas_name = match.group(1)
            # Capitalize first letter if needed
            atlas_name = atlas_name[0].upper() + atlas_name[1:]
            atlas_names.append(atlas_name)
    
    # Step 0: Make Table 1 for Demographic and Clinical Information
    # Extract variable names from the given statistical model string
    match = re.search(r'~(.*)\)', model_txt)
    if not match:
        raise ValueError("Invalid model string format. Expected '~' and ')'")
    
    model_vars = match.group(1)
    
    # Split by separators: +, -, (, |, *, :
    # First replace these with a common separator
    for sep in ['+', '-', '(', ')','|', '*', ':']:
        model_vars = model_vars.replace(sep, ' ')
    
    # Split by whitespace and remove empty strings
    var_list = [var.strip() for var in model_vars.split() if var.strip()]
    
    # Remove numeric values (like "1" in "(1|SITE)")
    var_list = [var for var in var_list if not var.isdigit()]
    
    # Load the CSV file as a DataFrame
    print(f"Loading subjects' infomation from {subjects_csv_file}...")
    df0 = pd.read_csv(subjects_csv_file)
    
    # Create a new DataFrame with only the specified columns
    df = df0[var_list].copy().dropna()
    
    # Make Table1 (by GROUP)
    df1= df.drop(table1_site_var,axis=1)
    table1 = TableOne(df1, groupby=table1_group_var, decimals=3, 
                    pval = True, htest_name=True,
                    dip_test=True, normal_test=True, tukey_test=True)
    table1.to_csv(os.path.join(output_dir,'table1.csv'))
    
    # Make TableS1 (by GROUP & SITE)
    tables1 = TableOne(df, groupby=table1_site_var, decimals=3)
    tables1.to_csv(os.path.join(output_dir,'tables1.csv'))
    
    # Step 1: Load correlation matrices
    print(f"Loading statistics matrix from: {stat_file}")
    stat_matrix = np.loadtxt(stat_file, delimiter=',')
    print(f"Loading p-values matrix from: {pval_file}")
    pval_matrix = np.loadtxt(pval_file, delimiter=',')
    
    # Step 2: Load node labels
    print(f"Loading node labels from: {labels_file}")
    labels_df = pd.read_csv(labels_file, delimiter='\t', header=None, names=['id', 'name'])
    
    # Process the node names: remove specific atlas prefixes and simplify hemisphere notation
    # Store raw node names for future reference
    raw_node_names = labels_df['name'].tolist()
    
    # Remove only specified atlas prefixes and simplify hemisphere notation
    clean_node_names = []
    for name in raw_node_names:
        # Remove only the specific atlas prefixes
        name = name.replace('Brainnetome_', '').replace('FreeSurfer_', '').replace('Buckner2011_17Networks_','Buckner_').replace('Power2011_','').replace('Schaefer2018_17Networks_','')
        # Replace Left- with L and Right- with R
        name = name.replace('Left-', 'L_').replace('Right-', 'R_').replace('LH_','L_').replace('RH_','R_')
        clean_node_names.append(name)
    
    node_names = clean_node_names
    
    # Step 3: Threshold the statistic matrix based on p-values
    print("Thresholding statistics matrix...")
    thresholded_matrix = np.zeros_like(stat_matrix)
    significant_mask = pval_matrix < pval_threshold
    thresholded_matrix[significant_mask] = stat_matrix[significant_mask]
    
    # Count significant connections (only upper triangle)
    significant_count = np.sum(significant_mask[np.triu_indices_from(significant_mask, k=1)])
    print(f"Found {significant_count} significant connections at p < {pval_threshold}")
    
    # Check if there are any significant connections
    has_significant_connections = significant_count > 0
    
    # Identify nodes that have at least one significant connection
    nodes_with_significant_connections = set()
    significant_nodes = []
    significant_node_names = []
    
    if has_significant_connections:
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if significant_mask[i, j]:
                    nodes_with_significant_connections.add(i)
                    nodes_with_significant_connections.add(j)
        
        # Convert set to list and sort by index
        significant_nodes = sorted(list(nodes_with_significant_connections))
        
        # Get names of nodes with significant connections
        significant_node_names = [node_names[i] for i in significant_nodes]
        
        # Extract submatrix with only significant nodes
        significant_thresholded_matrix = thresholded_matrix[np.ix_(significant_nodes, significant_nodes)]
    
    # Step 4: Create pandas DataFrames with node names as indices and columns
    stat_df = pd.DataFrame(stat_matrix, index=node_names, columns=node_names)
    pval_df = pd.DataFrame(pval_matrix, index=node_names, columns=node_names)
    thresholded_df = pd.DataFrame(thresholded_matrix, index=node_names, columns=node_names)
    
    # Step 5: Plot and save the thresholded matrix with vertical labels
    print("Plotting thresholded matrix...")
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    if has_significant_connections:
        # Decide which labels to use based on number of nodes
        matrix_display_labels = None
        matrix_display_matrix = None
        if len(node_names) > 20:
            # Only show labels for nodes with significant connections
            print(f"Using filtered labels for {len(significant_node_names)} nodes with significant connections in matrix plot")
            matrix_display_labels = significant_node_names
            matrix_display_matrix = significant_thresholded_matrix
        else:
            # Show all labels if 20 or fewer nodes
            print("Using all node labels for matrix plot")
            matrix_display_labels = node_names
            matrix_display_matrix = thresholded_matrix
        
        # Use nilearn's plot_matrix with vertical axis labels
        display = plotting.plot_matrix(
            matrix_display_matrix.T, 
            tri="lower",
            labels=matrix_display_labels, 
            vmax=abs(matrix_display_matrix).max(),
            vmin=-abs(matrix_display_matrix).max(),
            cmap='RdBu_r',
            colorbar=True
        )
    else:
        # Create a styled "No significant results" box similar to Table 2 style
        plt.text(0.5, 0.5, "No significant results.", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16,
                 transform=plt.gca().transAxes)
        # Add a light background and a border to match the table style
        plt.gca().add_patch(plt.Rectangle((0.25, 0.4), 0.5, 0.2, 
                                         facecolor='#f9f9f9', 
                                         edgecolor='#ddd',
                                         alpha=0.8,
                                         transform=plt.gca().transAxes))
        plt.axis('off')
    
    thresh_file_path = os.path.join(output_dir, "thresholded_matrix.png")
    plt.savefig(thresh_file_path, dpi=300, bbox_inches='tight')
    print(f"Saved thresholded matrix plot to: {thresh_file_path}")
    plt.close()
    
    # Step 6: Find node coordinates from atlas
    print("Finding node coordinates...")
    has_valid_coords = False
    node_coords = None
    
    if atlas_file and os.path.exists(atlas_file):
        print(f"Loading atlas file for node coordinates: {atlas_file}")
        try:
            # Load the atlas image
            atlas_img = nib.load(atlas_file)
            
            # Find coordinates for each region in the atlas
            print("Finding parcellation cut coordinates...")
            node_coords = find_parcellation_cut_coords(atlas_img)
            
            # Make sure we have the right number of coordinates
            if len(node_coords) == len(node_names):
                print(f"Successfully found {len(node_coords)} node coordinates from atlas file")
                has_valid_coords = True
            else:
                print(f"Warning: Number of coordinates ({len(node_coords)}) doesn't match number of nodes ({len(node_names)})")
                print("Cannot use these coordinates for visualization")
                node_coords = None
        except Exception as e:
            print(f"Error extracting coordinates from atlas: {e}")
            node_coords = None
    
    # Step 7: Plot connectome or brain template
    print("Plotting brain visualization...")
    plt.figure(figsize=(12, 12))
    
    if has_significant_connections and has_valid_coords:
        # Decide whether to filter nodes based on number
        use_filtered_nodes = len(node_names) > 20
        
        if use_filtered_nodes:
            print(f"Using filtered nodes for connectome plot: showing only {len(significant_nodes)} nodes with significant connections")
            
            # Extract the coordinates for only significant nodes
            significant_node_coords = np.array([node_coords[i] for i in significant_nodes])
            
            # Create a submatrix containing only the significant nodes 
            significant_matrix = thresholded_matrix[np.ix_(significant_nodes, significant_nodes)]
            
            # Plot connectome with only significant nodes
            connectome_display = plotting.plot_connectome(
                significant_matrix + significant_matrix.T,  # Make symmetric
                significant_node_coords,
                node_color='auto',
                title=None,
                colorbar=True,
                edge_cmap='RdBu_r',
                display_mode='ortho'
            )
        else:
            print("Using all nodes for connectome plot")
            # Plot connectome with all nodes
            connectome_display = plotting.plot_connectome(
                thresholded_matrix + thresholded_matrix.T,  # Make symmetric
                node_coords,
                node_color='auto',
                title=None,
                colorbar=True,
                edge_cmap='RdBu_r',
                display_mode='ortho'
            )
    else:
        # Create a styled "No significant results" box similar to Table 2 style
        plt.text(0.5, 0.5, "No significant results.", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16,
                 transform=plt.gca().transAxes)
        # Add a light background and a border to match the table style
        plt.gca().add_patch(plt.Rectangle((0.25, 0.4), 0.5, 0.2, 
                                         facecolor='#f9f9f9', 
                                         edgecolor='#ddd',
                                         alpha=0.8,
                                         transform=plt.gca().transAxes))
        plt.axis('off')
        print("No significant connections found. Displaying 'No significant results' message.")
    
    connectome_file_path = os.path.join(output_dir, "brain_connectome.png")
    plt.savefig(connectome_file_path, dpi=300, bbox_inches='tight')
    print(f"Saved brain visualization to: {connectome_file_path}")
    plt.close()
    
    # Step 8: Create and save tables
    print("Creating table of significant connections...")
    # Find significant connections (only upper triangle)
    significant_connections = []
    
    if has_significant_connections:
        for i in range(len(node_names)):
            for j in range(i+1, len(node_names)):  # Upper triangle only
                if significant_mask[i, j]:
                    significant_connections.append({
                        'Node1': node_names[i],
                        'Node2': node_names[j],
                        'Statistic': stat_matrix[i, j],
                        'P.fdr': pval_matrix[i, j]
                    })
    
    # Create DataFrame and sort by absolute statistic value
    sig_df = pd.DataFrame(significant_connections)
    if not sig_df.empty:
        sig_df['Abs_Stat'] = sig_df['Statistic'].abs()
        sig_df = sig_df.sort_values('Abs_Stat', ascending=False)
        sig_df = sig_df.drop('Abs_Stat', axis=1)
        
        # Save to CSV
        csv_file_path = os.path.join(output_dir, "significant_connections.csv")
        sig_df.to_csv(csv_file_path, index=False)
        print(f"Saved {len(sig_df)} significant connections to: {csv_file_path}")
    else:
        # Create empty DataFrame if no significant connections
        pd.DataFrame(columns=['Node1', 'Node2', 'Statistic', 'P.fdr']).to_csv(
            os.path.join(output_dir, "significant_connections.csv"), index=False)
        print("No significant connections found.")

    # Load table1 and tables1 for HTML display
    table1_html = ""
    tables1_html = ""
    
    # Load and format table1.csv to HTML
    table1_path = os.path.join(output_dir, 'table1.csv')
    if os.path.exists(table1_path):
        table1_df = pd.read_csv(table1_path, skiprows=1)
        # Replace NaN with empty string
        table1_df = table1_df.fillna("")
        # Rename columns
        if "Unnamed: 0" in table1_df.columns:
            table1_df = table1_df.rename(columns={"Unnamed: 0": "Variables"})
        if "Unnamed: 1" in table1_df.columns:
            table1_df = table1_df.rename(columns={"Unnamed: 1": ""})
        table1_html = table1_df.to_html(index=False, classes='table table-striped', border=0)
    
    # Load and format tables1.csv to HTML (skipping first row and transposing)
    tables1_path = os.path.join(output_dir, 'tables1.csv')
    if os.path.exists(tables1_path):
        # Skip the first row and transpose
        tables1_df = pd.read_csv(tables1_path, skiprows=1)
        # Replace NaN with empty string
        tables1_df = tables1_df.fillna("")
        # Rename columns
        if "Unnamed: 0" in tables1_df.columns:
            tables1_df = tables1_df.rename(columns={"Unnamed: 0": "SITE"})
        if "Unnamed: 1" in tables1_df.columns:
            tables1_df = tables1_df.rename(columns={"Unnamed: 1": ""})
        # Then use the first row as headers and drop that row
        tables1_df = tables1_df.transpose()
        tables1_df.columns = tables1_df.iloc[0]
        tables1_df = tables1_df.drop(tables1_df.index[0]) 
        tables1_html = tables1_df.to_html(index=True, classes='table table-striped', border=0)
 
    # Step 9: Generate HTML report
    print("Generating HTML report...")
    
    # Update descriptions based on whether we have significant connections
    if has_significant_connections:
        matrix_description = "Thresholded statistical matrix of interregional connectivity (p<sub>FDR</sub> &lt; {pval_threshold}). ".format(pval_threshold=pval_threshold)
        if len(node_names) > 20:
            matrix_description += "Only nodes with at least one significant connection are shown ({} out of {} total nodes). ".format(
                len(significant_node_names), len(node_names))
        
        matrix_description += "The lower triangular matrix displays significant connections, with color hue representing the direction of statistical effects " \
                             "and color intensity corresponding to the magnitude of test statistics. Non-significant connections are omitted."
        
        # Update connectome description based on whether we have valid coordinates
        if has_valid_coords:
            connectome_description = "Three-dimensional visualization of significant connections between brain regions (p<sub>FDR</sub> &lt; {pval_threshold}). ".format(pval_threshold=pval_threshold)
            if len(node_names) > 20:
                connectome_description += "Only nodes with at least one significant connection are shown ({} out of {} total nodes). ".format(
                    len(significant_node_names), len(node_names))
            
            connectome_description += "Edge colors represent the direction and strength of statistical associations between connected regions."
        else:
            connectome_description = "Brain connectome visualization could not be generated because valid coordinates could not be extracted from the atlas file."
    else:
        matrix_description = "No significant connections were detected at the threshold of p<sub>FDR</sub> &lt; {pval_threshold}.".format(pval_threshold=pval_threshold)
        connectome_description = "Glass brain visualization shown without connections, as no significant connections were detected at the threshold of p<sub>FDR</sub> &lt; {pval_threshold}.".format(pval_threshold=pval_threshold)
    
    # Update methods section with atlas names
    methods_paragraph = """<p>Statistical analysis was conducted using the Image-Based Mega- and Meta-Analysis (<a href="https://github.com/sundelinustc/IBMMA" target="_blank">IBMMA</a>) 
                framework for neuroimaging region-of-interest (ROI) connectivity analysis. 
                The analytical approach employed a comprehensive examination of functional connectivity patterns 
                between ROIs based on the """
    
    if atlas_names:
        if len(atlas_names) == 1:
            # Single atlas
            methods_paragraph += f"<i>{atlas_names[0]}</i> atlas parcellation.</p>"
        else:
            # Multiple atlases
            atlas_list = ", ".join([f"<i>{name}</i>" for name in atlas_names[:-1]])
            methods_paragraph += f"{atlas_list} and <i>{atlas_names[-1]}</i> atlas parcellations.</p>"
    else:
        methods_paragraph += "specified atlas parcellation.</p>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Correlation Matrix Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }}
            .caption {{
                margin-top: 5px;
                font-style: italic;
                color: #666;
            }}
            .methods {{
                text-align: justify;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #2c3e50;
            }}
            .pagination {{
                margin-top: 20px;
                text-align: center;
            }}
            .pagination a {{
                display: inline-block;
                padding: 8px 16px;
                text-decoration: none;
                color: #2c3e50;
                border: 1px solid #ddd;
                margin: 0 4px;
            }}
            .pagination a:hover {{
                background-color: #e9e9e9;
            }}
            .no-results {{
                text-align: center;
                padding: 20px;
                font-style: italic;
                color: #777;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Correlation Matrix Analysis Report</h1>
        
        <div class="section">
            <h2>Analysis Information</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Data Type</td>
                    <td>{data_type}</td>
                </tr>
                <tr>
                    <td>Model Number</td>
                    <td>{model_num}</td>
                </tr>
                <tr>
                    <td>Model Equation</td>
                    <td>{model_txt}</td>
                </tr>
                <tr>
                    <td>Effect of Interest</td>
                    <td>{effect_interest}</td>
                </tr>
                <tr>
                    <td>p<sub>FDR</sub> Threshold</td>
                    <td>{pval_threshold}</td>
                </tr>
                <tr>
                    <td>Number of Nodes</td>
                    <td>{len(node_names)}</td>
                </tr>
                <tr>
                    <td>Number of Significant Connections</td>
                    <td>{len(significant_connections)}</td>
                </tr>
                <tr>
                    <td>Number of Nodes with Significant Connections</td>
                    <td>{len(significant_node_names) if has_significant_connections else 0}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>File Paths</h2>
            <table>
                <tr>
                    <th>File Type</th>
                    <th>Path</th>
                </tr>
                <tr>
                    <td>Statistics File</td>
                    <td>{stat_file}</td>
                </tr>
                <tr>
                    <td>P-value File</td>
                    <td>{pval_file}</td>
                </tr>
                <tr>
                    <td>Node Labels File</td>
                    <td>{labels_file}</td>
                </tr>
                <tr>
                    <td>Atlas File</td>
                    <td>{atlas_file}</td>
                </tr>
                <tr>
                    <td>Output Directory</td>
                    <td>{output_dir}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Methods</h2>
            <div class="methods">
                {methods_paragraph}
                
                <p>All statistical comparisons were performed using two-tailed statistical tests with false discovery 
                rate (FDR) correction to account for multiple comparisons across the connection matrix. 
                Only connections that survived the statistical threshold of p<sub>FDR</sub> &lt; 0.05 were included 
                in subsequent analyses and visualizations, ensuring rigorous control of Type I error while 
                maintaining appropriate sensitivity to detect biologically meaningful effects.</p>
                
                <p>Visualization of the thresholded connection matrices and three-dimensional brain connectome 
                representations was implemented using the Nilearn package in Python. Matrix 
                visualizations were generated with a divergent color scheme to illustrate the directionality and 
                magnitude of significant associations, while connectome visualizations employed standardized 
                Montreal Neurological Institute (MNI) space coordinates to accurately represent the spatial 
                configuration of significant interregional connections.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Results</h2>
            
        <div class="section">
            <h2>Table 1. Demographic and Clinical Information</h2>
            <div class="table-responsive">
                {table1_html}
                <p><em>Note: Group comparisons of demographic and clinical variables.</em></p>
            </div>
        </div>
        
        <div class="section">
            <h2>Table S1. Site-Specific Information</h2>
            <div class="table-responsive">
                {tables1_html}
                <p><em>Note: Distribution of variables across different sites.</em></p>
            </div>
        </div>
            
            <h3>Thresholded Statistics Matrix of Connectivity</h3>
            <img src="thresholded_matrix.png" alt="Thresholded Statistics Matrix">
            <p class="caption">Figure 1: {matrix_description}</p>
            
            <h3>Brain Connectome</h3>
            <img src="brain_connectome.png" alt="Brain Connectome">
            <p class="caption">Figure 2: {connectome_description}</p>
        </div>
        
        <div class="section">
            <h2>Table 2. Significant Connections</h2>
    """
    
    # Add table of significant connections to HTML
    if has_significant_connections:
        html_content += """
            <table>
                <tr>
                    <th>#</th>
                    <th>Node 1</th>
                    <th>Node 2</th>
                    <th>Statistic</th>
                    <th><i>p</i><sub>FDR</sub></th>
                </tr>
        """
        
        # Add ALL connections to the table
        for idx, (_, row) in enumerate(sig_df.iterrows(), 1):
            html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{row['Node1']}</td>
                    <td>{row['Node2']}</td>
                    <td>{row['Statistic']:.3f}</td>
                    <td>{row['P.fdr']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            <p><i>Note: Table shows all significant connections ranked by absolute magnitude of test statistics. The complete set is also available as a CSV file: <a href="significant_connections.csv">significant_connections.csv</a>.</i></p>
        """
    else:
        html_content += """
            <div class="no-results">
                <p>No significant results.</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <footer style="margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 0.8em; color: #777; text-align: center;">
            Report generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """
        </footer>
    </body>
    </html>
    """
    
    # Write HTML report
    html_file_path = os.path.join(output_dir, "index.html")
    with open(html_file_path, 'w') as f:
        f.write(html_content)
    print(f"Saved HTML report to: {html_file_path}")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Open {html_file_path} to view the report.")
    

# Function to make HTML report for a single datatype, model, and effect of interest
def report_single(args):
    """
    Make HTML report for a single datatype, model, and effect of interest
    
    Args:
        report_dir (str): Path to the reports folder of a specified data type, e.g. "Reports/fALFF_falff"
        result_dir (str): Path to the results folder of a specified data type, e.g. "Results/fALFF_falff" and "Results/corrMatrix_atlas-brainnetome".
        model_name (str): Text of model name, e.g., "Model_01".
        model_formula (str): Text of modal formula, e.g. "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))".
        filter_string (str): Text string to filter the rows (subjects) of interest, e.g. "SEX==1, 20 < AGE < 60".
        effect_interest (str): Text of effect of interest, e.g., "GROUP" or "GROUP..AGE"
        model_Subjects (str): Path to the CSV file of datatype- & model-specific Subjects' info.
        labels_file (str): For connectome only. Path to the labels of atlas, e.g., "tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.txt"
        atlas_file (str): For connectome only. Path to the nifti of atlas, e.g., "tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.nii.gz"
        table1_site_var (str):  Text of the var name of site, e.g., "SITE"
        table1_group_var (str): Text of the var name of group, e.g., "GROUP"
    Outputs:
        Generate a single HTML report.
    """
    # Arguments
    report_dir, result_dir, model_name, model_formula, filter_string, effect_interest, model_Subjects, labels_file, atlas_file, table1_site_var, table1_group_var = args
    
    # Connectome
    try:
        # NIFTI or connectome through checking the representative file
        stat_csv   = os.path.join(result_dir, "Mega","TIDY","statistic",effect_interest,"OUT_" + model_name + ".csv")
        stat_nifti = os.path.join(result_dir, "Mega","TIDY","statistic",effect_interest,"OUT_" + model_name + ".nii.gz_pTFCE","pTFCE-z-score-map.nii.gz")
        
        if os.path.isfile(stat_csv):
            generate_connectome_report(
                subjects_csv_file=model_Subjects,
                stat_file=os.path.join(result_dir, "Mega", "TIDY", "statistic",   effect_interest, "OUT_" + model_name + ".csv"), # Path to the statistics matrix file
                pval_file=os.path.join(result_dir, "Mega", "TIDY", "p.value.fdr", effect_interest, "OUT_" + model_name + ".csv"), # Path to the FDR-corrected p-values matrix file
                labels_file=labels_file, # Input needed
                atlas_file=atlas_file,   # Input needed
                pval_threshold=0.05,
                output_dir=os.path.join(report_dir, "Mega", effect_interest,model_name),  # Output directory
                data_type=os.path.basename(result_dir), # get datatype though the folder name
                model_num=model_name,
                model_txt=model_formula,
                effect_interest=effect_interest,
                table1_site_var=table1_site_var, 
                table1_group_var=table1_group_var
            )
        elif os.path.isfile(stat_nifti):
            generate_nifti_report(
                subjects_csv_file=model_Subjects,
                pos_map_path=  os.path.join(result_dir, "Mega","TIDY","statistic",effect_interest,"OUT_" + model_name + ".nii.gz_pTFCE", "pTFCE-z-score-map.nii.gz"),           # pTFCE pos results
                neg_map_path=  os.path.join(result_dir, "Mega","TIDY","statistic",effect_interest,"OUT_" + model_name + ".nii.gz_pTFCE", "pTFCE-z-reversed-score-map.nii.gz"),  # pTFCE neg results
                thres_txt_path=os.path.join(result_dir, "Mega","TIDY","statistic",effect_interest,"OUT_" + model_name + ".nii.gz_pTFCE", "thres_z_fwer0.05.txt"), # text file containing Z value coresponding to FWE corrected p = 0.05 for both pos and neg pTFCE images
                output_dir=os.path.join(report_dir, "Mega", effect_interest,model_name),  # Output directory
                data_type=os.path.basename(result_dir), # get datatype though the folder name
                model_num=model_name,
                model_txt=model_formula,
                effect_interest=effect_interest,
                peak_region_min_percentage=10,
                cluster_region_min_volume=10, 
                table1_site_var=table1_site_var, 
                table1_group_var=table1_group_var
            )
        else:
            pass
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

# Class for HTML_Report
class HTML_Report:
    def __init__(self, num_processes=None):
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count() # use all CPUs if not specified
          
    def report(self, report_dir, result_dir, model_name, model_formula, filter_string, model_Subjects, atlas, labels, table1_site_var, table1_group_var):
        """
        Generate HTML reports of statistical outputs for voxel-wised (nifti) or atlas-based connectome (matrix).
        
        Args:
            report_dir (str): Path to the reports folder of a specified data type, e.g. "Reports/fALFF_falff".
            result_dir (str): Path to the results folder of a specified data type, e.g. "Results/fALFF_falff" and "Results/corrMatrix_atlas-brainnetome".
            model_name (str): Text of model name, e.g., "Model_01".
            model_formula (str): Text of modal formula, e.g. "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))".
            filter_string (str): Text string to filter the rows (subjects) of interest, e.g. "SEX==1, 20 < AGE < 60".
            model_Subjects (str): Path to the CSV file of datatype- & model-specific Subjects' info.
            atlas (str):  For atlas-based connectome only. Path to the nifti of atlas, e.g. "tpl-MNI152NLin2009cAsym_atlas-brainnetome_dseg.nii.gz"
            labels (str): For atlas-based connectome only. Txt file of labels of the given atlas, e.g. "tpl-MNI152NLin2009cAsym_atlas-brainnetome_dseg.txt"
            table1_site_var (str):  Text of var name for site, e.g., "SITE" 
            table1_group_var (str): Text of var name for group, e.g., "GROUP"
        
        Outputs:
            Generate HTML reports.
        """
        # print information
        t0 = time.time() # start time
        print(f"\nHTML Reports ... ")
        
        # Parameters
        data_type = os.path.basename(result_dir) # e.g. "corrMatrix_atlas-brainnetome" or "corrMatrix_atlas-freeSurfer"
        model_num = model_name # e.g., "Model_01"
        model_txt = model_formula # "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))"
        
        # Extract variables from model texts
        model_variables     = extract_model_variables(model_formula) #  fixed factors
        model_variables_all = extract_model_variables(model_formula, random_factor_included=True) # fixed + random factors
        random_var = list(set(model_variables_all) - set(model_variables))[0] # random variable, e.g., 'SITE'
        
        # Find categorical variables
        list_cat_var = find_categorical_columns(model_Subjects, columns_to_check=model_variables) 
        # The first categorical variable in the above list will be recognized as the GROUP factor to be reported in Table 1 and Table S1
        group_var = list_cat_var[0] # e.g., "GROUP"
        
        # Search for effects of interest: folder name of the corresponding models
        list_effect_interest = [os.path.basename(root) 
                 for root, _, files in os.walk(os.path.join(result_dir,'Mega','TIDY','statistic')) 
                 for file in files 
                 if file == model_name + '.csv' and not os.path.basename(root).startswith('sd__')]
            
        # Arguments
        my_args = [(report_dir, result_dir, model_name, model_formula, filter_string, effect_interest, model_Subjects, labels, atlas, table1_site_var, table1_group_var) for effect_interest in list_effect_interest]
        
        # # For test purpose only !!
        # report_single(my_args[0])
        
        # Parallel processing
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(report_single, my_args)
            
        # print ending info
        print(f"HTML Reports completed!\nTime elapsed (in secs): {time.time()-t0}\n")
