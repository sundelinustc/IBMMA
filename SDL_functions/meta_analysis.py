import os
import re
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import time # for measuring time elapsed

# function (1.1)
def extract_zmap_data(folder_path, debug=False):
    """
    Recursively search for Zmap.nii.gz files and extract information into a DataFrame.
    
    Parameters:
    -----------
    folder_path : str
        The root folder path to search recursively for Zmap.nii.gz files
    debug : bool
        If True, print debugging information
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: model_name, site, contrast, stat_map
    """
    
    # Initialize list to store results
    results = []
    
    # Convert to Path object for easier handling
    root_path = Path(folder_path)
    
    # Check if the folder exists
    if not root_path.exists():
        print(f"Error: Folder {folder_path} does not exist!")
        return pd.DataFrame()
    
    if debug:
        print(f"Searching in: {root_path}")
        print(f"Folder exists: {root_path.exists()}")
        print(f"Is directory: {root_path.is_dir()}")
    
    # Try different search patterns
    search_patterns = [
        "*Zmap.nii.gz",
        "*zmap.nii.gz", 
        "**/*Zmap.nii.gz",
        "**/*zmap.nii.gz",
        "*.nii.gz",
        "**/*.nii.gz"
    ]
    
    zmap_files = []
    
    for pattern in search_patterns:
        files = list(root_path.rglob(pattern))
        if debug:
            print(f"Pattern '{pattern}' found {len(files)} files")
            if files and len(files) <= 5:  # Show first few files
                for f in files[:5]:
                    print(f"  - {f}")
        
        # Filter for files containing "Zmap" (case-insensitive)
        if "zmap" in pattern.lower():
            zmap_files.extend(files)
        else:
            # For general .nii.gz files, filter by name
            zmap_candidates = [f for f in files if "zmap" in f.name.lower()]
            zmap_files.extend(zmap_candidates)
            if debug and zmap_candidates:
                print(f"  Found {len(zmap_candidates)} Zmap files from general search")
        
        if zmap_files:
            break  # Stop if we found files
    
    if debug:
        print(f"Total Zmap files found: {len(zmap_files)}")
    
    # Remove duplicates
    zmap_files = list(set(zmap_files))
    
    for file_path in zmap_files:
        # Convert to string for regex operations
        full_path = str(file_path)
        
        # Extract contrast: characters between "statistic" and "OUT_"
        contrast_match = re.search(r'statistic(.*?)OUT_', full_path)
        contrast = contrast_match.group(1) if contrast_match else None
        # Remove path separators from contrast
        if contrast:
            contrast = contrast.replace('/', '').replace('\\', '')
        
        # Extract model_name: characters between "OUT_" and ".nii.gz-pTFCE"
        model_match = re.search(r'OUT_(.*?)\.nii\.gz-pTFCE', full_path)
        if not model_match:
            # Alternative pattern if "-pTFCE" is not present
            model_match = re.search(r'OUT_(.*?)\.nii\.gz', full_path)
        model_name = model_match.group(1) if model_match else None
        
        # Extract site: characters between "Meta" and "TIDY"
        site_match = re.search(r'Meta(.*?)TIDY', full_path)
        site = site_match.group(1) if site_match else None
        # Remove path separators from site
        if site:
            site = site.replace('/', '').replace('\\', '')
        
        # Create row dictionary with specified column order
        row = {
            'model_name': model_name,
            'site': site,
            'contrast': contrast,
            'stat_map': full_path
        }
        
        results.append(row)
    
    # Create DataFrame with specified column order
    df = pd.DataFrame(results, columns=['model_name', 'site', 'contrast', 'stat_map'])
    
    return df

# function (1.2)
def extract_sample_sizes(csv_file_path):
    """
    Read 2nd and 3rd rows from CSV, remove columns up to and including "Overall",
    and create a DataFrame with site names and sample sizes.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: site, n
    """
    
    # Read the CSV file without headers to get raw data
    df_raw = pd.read_csv(csv_file_path, header=None)
    
    # Extract the 2nd row (index 1) and 3rd row (index 2)
    upper_row = df_raw.iloc[1].tolist()  # 2nd row - site names
    lower_row = df_raw.iloc[2].tolist()  # 3rd row - sample sizes
    
    # Find the index of "Overall" column
    try:
        overall_index = upper_row.index("Overall")
    except ValueError:
        # If "Overall" not found, try case-insensitive search
        overall_index = None
        for i, val in enumerate(upper_row):
            if str(val).lower() == "overall":
                overall_index = i
                break
        
        if overall_index is None:
            print("Warning: 'Overall' column not found. Using all columns.")
            overall_index = -1  # Use all columns if "Overall" not found
    
    # Remove all columns including and before "Overall"
    if overall_index >= 0:
        sites = upper_row[overall_index + 1:]  # Keep columns after "Overall"
        sample_sizes = lower_row[overall_index + 1:]  # Corresponding sample sizes
    else:
        sites = upper_row
        sample_sizes = lower_row
    
    # Create DataFrame
    df = pd.DataFrame({
        'site': sites,
        'n': sample_sizes
    })
    
    # Clean up the data - remove any NaN or empty values
    df = df.dropna()
    df = df[df['site'].astype(str).str.strip() != '']
    df = df[df['site'].astype(str) != 'nan']
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

# Function (2.1) - Load and validate images
def load_images(image_paths: Dict[str, Union[str, Path]], 
               mask_path: Optional[Union[str, Path]] = None,
               sample_sizes: Optional[Dict[str, int]] = None):
    """
    Load NIfTI images from multiple sites.
    
    Parameters:
    -----------
    image_paths : dict
        Dictionary mapping site names to image file paths
    mask_path : str or Path, optional
        Path to brain mask image
    sample_sizes : dict, optional
        Dictionary mapping site names to sample sizes (number of subjects)
    
    Returns:
    --------
    tuple : (site_images, site_data, template_img, sample_sizes, brain_shape)
    """
    print("Loading images...")
    
    site_images = {}
    site_data = {}
    template_img = None
    brain_shape = None
    
    # Store sample sizes
    if sample_sizes is None:
        sample_sizes = {}
    
    for site_name, img_path in image_paths.items():
        try:
            img = nib.load(img_path)
            site_images[site_name] = img
            site_data[site_name] = img.get_fdata()
            
            # Set default sample size if not provided
            if site_name not in sample_sizes:
                sample_sizes[site_name] = 1
                print(f"  Warning: No sample size provided for {site_name}, using default n=1")
            
            # Set template image (use first image as template)
            if template_img is None:
                template_img = img
                brain_shape = site_data[site_name].shape
            
            print(f"  Loaded {site_name}: {img_path} (n={sample_sizes[site_name]})")
            
        except Exception as e:
            print(f"  Error loading {site_name}: {e}")
            continue
    
    n_sites = len(site_data)
    print(f"Successfully loaded {n_sites} images")
    print(f"Total sample size: {sum(sample_sizes.values())}")
    
    return site_images, site_data, template_img, sample_sizes, brain_shape

# Function (2.2) - Create brain mask
def create_brain_mask(site_data: Dict[str, np.ndarray], 
                     mask_path: Optional[Union[str, Path]] = None,
                     brain_shape: Tuple = None):
    """
    Load brain mask or create a default one.
    
    Parameters:
    -----------
    site_data : dict
        Dictionary of site data arrays
    mask_path : str or Path, optional
        Path to brain mask image
    brain_shape : tuple
        Shape of brain images
    
    Returns:
    --------
    np.ndarray : Brain mask as boolean array
    """
    if mask_path:
        try:
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata().astype(bool)
            print(f"Loaded brain mask: {mask_path}")
            return mask
        except Exception as e:
            print(f"Error loading mask: {e}. Creating default mask...")
    
    # Create default mask
    print("Creating default brain mask...")
    
    # Initialize mask with True values
    mask = np.ones(brain_shape, dtype=bool)
    
    # Exclude voxels that are zero or NaN in any image
    for site_name, data in site_data.items():
        valid_voxels = ~(np.isnan(data) | np.isinf(data) | (data == 0))
        mask = mask & valid_voxels
    
    n_voxels = np.sum(mask)
    print(f"Default mask created with {n_voxels} valid voxels")
    
    return mask

# Function (3.1) - Stouffer's method
def stouffers_method(site_data: Dict[str, np.ndarray], 
                    mask: np.ndarray,
                    brain_shape: Tuple,
                    sample_sizes: Dict[str, int],
                    weighted: bool = True) -> Dict[str, np.ndarray]:
    """
    Perform Stouffer's method for combining z-scores.
    
    Parameters:
    -----------
    site_data : dict
        Dictionary of site data arrays
    mask : np.ndarray
        Brain mask
    brain_shape : tuple
        Shape of brain images
    sample_sizes : dict
        Sample sizes for each site
    weighted : bool
        Whether to use sample size weights
    
    Returns:
    --------
    dict : Dictionary containing Stouffer's method results
    """
    print(f"Performing Stouffer's method meta-analysis (weighted={weighted})...")
    
    # Stack all z-values
    z_stack = np.stack([data for data in site_data.values()], axis=-1)
    
    # Apply mask
    masked_z = z_stack[mask]
    
    site_names = list(site_data.keys())
    n_sites = len(site_names)
    
    if weighted and all(sample_sizes[site] > 1 for site in site_names):
        # Weighted Stouffer's method
        weights = np.array([sample_sizes[site] for site in site_names])
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted combination
        stouffer_z = np.nansum(masked_z * weights[np.newaxis, :], axis=1) / np.sqrt(np.sum(weights**2))
        
        print(f"  Using weighted Stouffer's with weights: {dict(zip(site_names, weights))}")
    else:
        # Unweighted Stouffer's method
        stouffer_z = np.nansum(masked_z, axis=1) / np.sqrt(n_sites)
        print("  Using unweighted Stouffer's method")
    
    # P-values from combined z-scores
    stouffer_p = 2 * (1 - stats.norm.cdf(np.abs(stouffer_z)))
    
    # Standard error (theoretical)
    stouffer_se = np.ones_like(stouffer_z)  # SE = 1 for standardized combined z
    
    # Create full brain arrays
    results = {}
    for name, values in [('z_score', stouffer_z), ('std_error', stouffer_se), 
                       ('p_value', stouffer_p)]:
        full_array = np.zeros(brain_shape)
        full_array[mask] = values
        results[name] = full_array
    
    # Add sample size map
    n_valid = np.sum(~np.isnan(masked_z), axis=1)
    n_array = np.zeros(brain_shape)
    n_array[mask] = n_valid
    results['n_sites'] = n_array
    
    # Add total sample size map
    total_n = np.sum([sample_sizes[site] for site in site_names])
    total_n_array = np.full(brain_shape, total_n)
    results['total_sample_size'] = total_n_array
    
    print("Stouffer's method completed")
    
    return results

# Function (3.2) - Fisher's method
def fishers_method(site_data: Dict[str, np.ndarray], 
                  mask: np.ndarray,
                  brain_shape: Tuple) -> Dict[str, np.ndarray]:
    """
    Perform Fisher's method for combining p-values.
    
    Parameters:
    -----------
    site_data : dict
        Dictionary of site data arrays
    mask : np.ndarray
        Brain mask
    brain_shape : tuple
        Shape of brain images
    
    Returns:
    --------
    dict : Dictionary containing Fisher's method results
    """
    print("Performing Fisher's method meta-analysis...")
    
    # Stack all z-values and convert to p-values
    z_stack = np.stack([data for data in site_data.values()], axis=-1)
    masked_z = z_stack[mask]
    
    # Convert z-scores to two-tailed p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(masked_z)))
    
    # Fisher's method: -2 * sum(ln(p_i)) ~ Chi-square(2k)
    # Handle zero p-values by setting minimum p-value
    min_p = 1e-16
    p_values = np.maximum(p_values, min_p)
    
    # Calculate Fisher's statistic
    fisher_stat = -2 * np.nansum(np.log(p_values), axis=1)
    
    # Degrees of freedom (2 * number of studies)
    df = 2 * np.sum(~np.isnan(p_values), axis=1)
    
    # P-values from chi-square distribution
    fisher_p = 1 - stats.chi2.cdf(fisher_stat, df)
    
    # Convert back to z-scores for consistency
    fisher_z = stats.norm.ppf(1 - fisher_p/2) * np.sign(np.nanmean(masked_z, axis=1))
    
    # Standard error (not directly applicable for Fisher's method)
    fisher_se = np.ones_like(fisher_z)  # Placeholder
    
    # Create full brain arrays
    results = {}
    for name, values in [('z_score', fisher_z), ('std_error', fisher_se), 
                       ('p_value', fisher_p), ('chi2_statistic', fisher_stat)]:
        full_array = np.zeros(brain_shape)
        full_array[mask] = values
        results[name] = full_array
    
    # Add degrees of freedom map
    df_array = np.zeros(brain_shape)
    df_array[mask] = df
    results['degrees_freedom'] = df_array
    
    # Add sample size map
    n_valid = np.sum(~np.isnan(p_values), axis=1)
    n_array = np.zeros(brain_shape)
    n_array[mask] = n_valid
    results['n_sites'] = n_array
    
    print("Fisher's method completed")
    
    return results

# Function (4.1) - Jackknife analysis
def jackknife_analysis(site_data: Dict[str, np.ndarray],
                      mask: np.ndarray,
                      brain_shape: Tuple,
                      method: str = 'stouffers') -> Dict[str, Dict[str, np.ndarray]]:
    """
    Perform jackknife analysis to evaluate site contributions.
    
    Parameters:
    -----------
    site_data : dict
        Dictionary of site data arrays
    mask : np.ndarray
        Brain mask
    brain_shape : tuple
        Shape of brain images
    method : str
        Which meta-analysis method to use ('stouffers' or 'fishers')
    
    Returns:
    --------
    dict : Dictionary containing jackknife results for each site
    """
    print(f"Performing jackknife analysis using {method} method...")
    
    site_names = list(site_data.keys())
    n_sites = len(site_names)
    z_stack = np.stack([site_data[name] for name in site_names], axis=-1)
    masked_z = z_stack[mask]
    
    jackknife_results = {}
    
    for i, excluded_site in enumerate(site_names):
        print(f"  Jackknife iteration: excluding {excluded_site}")
        
        # Create jackknife sample (exclude one site)
        jk_indices = [j for j in range(n_sites) if j != i]
        jk_z = masked_z[:, jk_indices]
        
        # Apply the selected meta-analysis method on jackknife sample
        if method == 'stouffers':
            # Stouffer's method
            jk_combined_z = np.nansum(jk_z, axis=1) / np.sqrt(len(jk_indices))
            jk_p = 2 * (1 - stats.norm.cdf(np.abs(jk_combined_z)))
            jk_se = np.ones_like(jk_combined_z)
            
            results_list = [('z_score', jk_combined_z), ('std_error', jk_se), ('p_value', jk_p)]
            
        elif method == 'fishers':
            # Fisher's method
            jk_p_values = 2 * (1 - stats.norm.cdf(np.abs(jk_z)))
            min_p = 1e-16
            jk_p_values = np.maximum(jk_p_values, min_p)
            
            fisher_stat = -2 * np.nansum(np.log(jk_p_values), axis=1)
            df = 2 * np.sum(~np.isnan(jk_p_values), axis=1)
            jk_combined_p = 1 - stats.chi2.cdf(fisher_stat, df)
            jk_combined_z = stats.norm.ppf(1 - jk_combined_p/2) * np.sign(np.nanmean(jk_z, axis=1))
            jk_se = np.ones_like(jk_combined_z)
            
            results_list = [('z_score', jk_combined_z), ('std_error', jk_se), 
                          ('p_value', jk_combined_p), ('chi2_statistic', fisher_stat)]
        
        # Store results with method prefix
        jk_results = {}
        for name, values in results_list:
            full_array = np.zeros(brain_shape)
            full_array[mask] = values
            jk_results[f'stouffer_{name}'] = full_array
        
        jackknife_results[excluded_site] = jk_results
    
    print("Jackknife analysis completed")
    
    return jackknife_results

# Function (4.2) - Calculate site contributions
def calculate_site_contributions(site_data: Dict[str, np.ndarray],
                               meta_results: Dict[str, np.ndarray],
                               jackknife_results: Dict[str, Dict[str, np.ndarray]],
                               mask: np.ndarray,
                               method: str = 'stouffers') -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate the influence of each site on the meta-analysis results.
    
    Parameters:
    -----------
    site_data : dict
        Dictionary of site data arrays
    meta_results : dict
        Meta-analysis results
    jackknife_results : dict
        Jackknife analysis results
    mask : np.ndarray
        Brain mask
    method : str
        Which meta-analysis method to use for contributions
    
    Returns:
    --------
    dict : Dictionary containing contribution measures for each site
    """
    print(f"Calculating site contributions using {method} method...")
    
    if not meta_results or not jackknife_results:
        raise ValueError("Must run meta-analysis and jackknife analysis first")
    
    # Get the appropriate z-score results
    z_key = f'stouffer_z_score'
    if z_key not in meta_results:
        raise ValueError(f"No results found for method: {method}. Available keys: {list(meta_results.keys())}")
    
    contributions = {}
    full_z = meta_results[z_key]
    n_sites = len(site_data)
    
    for site_name in site_data.keys():
        jk_z = jackknife_results[site_name][z_key]
        
        # Influence measure: difference between full and jackknife estimates
        influence = (n_sites - 1) * (full_z - jk_z)
        
        # Standardized influence
        std_influence = influence / np.nanstd(full_z[mask])
        
        contributions[site_name] = {
            'influence': influence,
            'standardized_influence': std_influence,
            'absolute_influence': np.abs(influence)
        }
    
    print("Site contributions calculated")
    
    return contributions

# Function (5.1) - Save NIfTI results
def save_nifti_results(results_dict: Dict[str, np.ndarray], 
                      template_img: nib.Nifti1Image,
                      output_path: Path,
                      prefix: str = "meta"):
    """Save analysis results as NIfTI images."""
    print(f"Saving NIfTI results with prefix '{prefix}'...")
    
    results_dir = output_path / "nifti_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in results_dict.items():
        filename = f"{prefix}_{name}.nii.gz"
        filepath = results_dir / filename
        
        # Create NIfTI image with same header as template
        result_img = nib.Nifti1Image(data, template_img.affine, 
                                   template_img.header)
        nib.save(result_img, str(filepath))
        print(f"  Saved: {filename}")

# Function (5.2) - Save summary statistics
def save_summary_statistics(site_data: Dict[str, np.ndarray],
                          sample_sizes: Dict[str, int],
                          meta_results: Dict[str, np.ndarray],
                          mask: np.ndarray,
                          output_path: Path):
    """Save summary statistics and site information."""
    print("Saving summary statistics...")
    
    stats_dir = output_path / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Site summary
    site_summary = []
    for site_name, data in site_data.items():
        masked_data = data[mask]
        site_info = {
            'site': site_name,
            'sample_size': sample_sizes.get(site_name, 1),
            'n_voxels': np.sum(mask),
            'mean_z': np.nanmean(masked_data),
            'std_z': np.nanstd(masked_data),
            'min_z': np.nanmin(masked_data),
            'max_z': np.nanmax(masked_data),
            'n_positive': np.sum(masked_data > 0),
            'n_negative': np.sum(masked_data < 0)
        }
        site_summary.append(site_info)
    
    df_sites = pd.DataFrame(site_summary)
    df_sites.to_csv(stats_dir / "site_summary.csv", index=False)
    
    # Meta-analysis summary
    if meta_results:
        # Use Stouffer's method results for summary
        meta_z = meta_results['stouffer_z_score'][mask]
        meta_p = meta_results['stouffer_p_value'][mask]
        
        meta_summary = {
            'analysis_date': datetime.now().isoformat(),
            'n_sites': len(site_data),
            'total_sample_size': sum(sample_sizes.values()),
            'mean_sample_size': np.mean(list(sample_sizes.values())),
            'n_voxels': np.sum(mask),
            'meta_mean_z': np.nanmean(meta_z),
            'meta_std_z': np.nanstd(meta_z),
            'n_significant_p001': np.sum(meta_p < 0.001),
            'n_significant_p01': np.sum(meta_p < 0.01),
            'n_significant_p05': np.sum(meta_p < 0.05),
            'site_names': ', '.join(list(site_data.keys())),
            'sample_sizes': ', '.join([f"{site}:{n}" for site, n in sample_sizes.items()])
        }
        
        # Convert to DataFrame and save as CSV
        df_meta = pd.DataFrame([meta_summary])
        df_meta.to_csv(stats_dir / "meta_analysis_summary.csv", index=False)
    
    print("Summary statistics saved")

# Function (5.3) - Create visualizations
def create_visualizations(site_data: Dict[str, np.ndarray],
                        meta_results: Dict[str, np.ndarray],
                        jackknife_results: Dict[str, Dict[str, np.ndarray]],
                        site_contributions: Dict[str, Dict[str, np.ndarray]],
                        mask: np.ndarray,
                        output_path: Path):
    """Create visualization plots."""
    print("Creating visualizations...")
    
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Site z-score distributions
    plt.figure(figsize=(12, 8))
    site_z_values = []
    site_labels = []
    
    for site_name, data in site_data.items():
        masked_data = data[mask]
        site_z_values.extend(masked_data[~np.isnan(masked_data)])
        site_labels.extend([site_name] * len(masked_data[~np.isnan(masked_data)]))
    
    df_plot = pd.DataFrame({'z_score': site_z_values, 'site': site_labels})
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_plot, x='site', y='z_score')
    plt.xticks(rotation=45)
    plt.title('Z-score Distributions by Site')
    
    # 2. Skip meta-analysis histogram (not needed)
    
    # 3. Site influence plot
    if site_contributions:
        plt.subplot(2, 2, 3)
        influences = []
        sites = []
        
        for site_name, contrib in site_contributions.items():
            inf_values = contrib['absolute_influence'][mask]
            influences.append(np.nanmean(inf_values))
            sites.append(site_name)
        
        plt.bar(sites, influences)
        plt.xticks(rotation=45)
        plt.ylabel('Mean Absolute Influence')
        plt.title('Site Influence on Meta-analysis')
    
    # 4. Jackknife stability
    if jackknife_results:
        plt.subplot(2, 2, 4)
        jk_means = []
        
        for site_name, jk_results_site in jackknife_results.items():
            jk_z = jk_results_site['stouffer_z_score'][mask]
            jk_means.append(np.nanmean(jk_z))
        
        plt.plot(range(len(jk_means)), jk_means, 'o-')
        plt.axhline(y=np.nanmean(meta_results['stouffer_z_score'][mask]), 
                   color='r', linestyle='--', label='Full sample')
        plt.xlabel('Jackknife Sample')
        plt.ylabel('Mean Z-score')
        plt.title('Jackknife Stability')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "meta_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved")

class Meta:
    """
    A comprehensive class for performing image-based meta-analysis on NIfTI brain images
    with jackknife methods to evaluate site contribution.
    """
    
    def __init__(self):
        """
        Initialize the meta-analysis object.
        """
        # Data storage
        self.site_images = {}  # Dict: site_name -> nibabel image
        self.site_data = {}    # Dict: site_name -> numpy array
        self.mask = None       # Brain mask
        self.template_img = None  # Template image for saving results
        
        # Results storage
        self.meta_results = {}
        self.jackknife_results = {}
        self.site_contributions = {}
        
        # Analysis parameters
        self.n_sites = 0
        self.brain_shape = None
        self.sample_sizes = {}
        
        print("Meta-analysis initialized.")
    
    def load_images(self, image_paths: Dict[str, Union[str, Path]], 
                   mask_path: Optional[Union[str, Path]] = None,
                   sample_sizes: Optional[Dict[str, int]] = None):
        """
        Load NIfTI images from multiple sites.
        
        Parameters:
        -----------
        image_paths : dict
            Dictionary mapping site names to image file paths
        mask_path : str or Path, optional
            Path to brain mask image
        sample_sizes : dict, optional
            Dictionary mapping site names to sample sizes (number of subjects)
        """
        # Use the external load_images function
        self.site_images, self.site_data, self.template_img, self.sample_sizes, self.brain_shape = \
            load_images(image_paths, mask_path, sample_sizes)
        
        self.n_sites = len(self.site_data)
        
        # Create brain mask
        self.mask = create_brain_mask(self.site_data, mask_path, self.brain_shape)
    
    def stouffers_method(self, weighted: bool = True) -> Dict[str, np.ndarray]:
        """
        Perform Stouffer's method for combining z-scores.
        
        Parameters:
        -----------
        weighted : bool
            Whether to use sample size weights (default: True if sample sizes provided)
        
        Returns:
        --------
        dict : Dictionary containing Stouffer's method results
        """
        results = stouffers_method(self.site_data, self.mask, self.brain_shape, 
                                 self.sample_sizes, weighted)
        
        self.meta_results.update({f'stouffer_{k}': v for k, v in results.items()})
        return results
    
    def fishers_method(self) -> Dict[str, np.ndarray]:
        """
        Perform Fisher's method for combining p-values.
        
        Returns:
        --------
        dict : Dictionary containing Fisher's method results
        """
        results = fishers_method(self.site_data, self.mask, self.brain_shape)
        
        self.meta_results.update({f'fisher_{k}': v for k, v in results.items()})
        return results
    
    def jackknife_analysis(self, method='stouffers') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Perform jackknife analysis to evaluate site contributions.
        
        Parameters:
        -----------
        method : str
            Which meta-analysis method to use ('stouffers' or 'fishers')
        
        Returns:
        --------
        dict : Dictionary containing jackknife results for each site
        """
        self.jackknife_results = jackknife_analysis(self.site_data, self.mask, 
                                                   self.brain_shape, method)
        return self.jackknife_results
    
    def calculate_site_contributions(self, method='stouffers') -> Dict[str, np.ndarray]:
        """
        Calculate the influence of each site on the meta-analysis results.
        
        Parameters:
        -----------
        method : str
            Which meta-analysis method to use for contributions ('stouffers' or 'fishers')
        
        Returns:
        --------
        dict : Dictionary containing contribution measures for each site
        """
        self.site_contributions = calculate_site_contributions(
            self.site_data, self.meta_results, self.jackknife_results, 
            self.mask, method)
        
        return self.site_contributions
    
    def save_nifti_results(self, results_dict: Dict[str, np.ndarray], 
                          prefix: str = "meta", output_path: Path = None):
        """Save analysis results as NIfTI images."""
        if output_path is None:
            raise ValueError("output_path must be provided")
        save_nifti_results(results_dict, self.template_img, output_path, prefix)
    
    def save_summary_statistics(self, output_path: Path = None):
        """Save summary statistics and site information."""
        if output_path is None:
            raise ValueError("output_path must be provided")
        save_summary_statistics(self.site_data, self.sample_sizes, self.meta_results, 
                               self.mask, output_path)
    
    def create_visualizations(self, output_path: Path = None):
        """Create visualization plots."""
        if output_path is None:
            raise ValueError("output_path must be provided")
        create_visualizations(self.site_data, self.meta_results, self.jackknife_results,
                            self.site_contributions, self.mask, output_path)
    
    def run_complete_analysis(self, output_path: Union[str, Path], save_all: bool = True) -> Dict:
        """
        Run the complete meta-analysis pipeline.
        
        Parameters:
        -----------
        output_path : str or Path
            Directory path where all results will be saved
        save_all : bool
            Whether to save all results to disk
        
        Returns:
        --------
        dict : Dictionary containing all analysis results
        """
        print("Running complete meta-analysis pipeline...")
        
        # Convert output_path to Path object
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if images are loaded
        if not self.site_data:
            raise ValueError("No images loaded. Call load_images() first.")
        
        # Run analyses
        results = {}
        
        # Stouffer's method meta-analysis
        results['stouffers'] = self.stouffers_method()
        
        # Fisher's method meta-analysis
        results['fishers'] = self.fishers_method()
        
        # Jackknife analysis (using Stouffer's method)
        results['jackknife'] = self.jackknife_analysis('stouffers')
        
        # Site contributions (using Stouffer's method)
        results['contributions'] = self.calculate_site_contributions('stouffers')
        
        if save_all:
            # Save NIfTI results
            self.save_nifti_results(results['stouffers'], "stouffers", output_path)
            self.save_nifti_results(results['fishers'], "fishers", output_path)
            
            # Save jackknife results
            for site_name, jk_results in results['jackknife'].items():
                self.save_nifti_results(jk_results, f"jackknife_{site_name}", output_path)
            
            # Save contribution maps
            for site_name, contrib in results['contributions'].items():
                self.save_nifti_results(contrib, f"contribution_{site_name}", output_path)
            
            # Save summary statistics and plots
            self.save_summary_statistics(output_path)
            self.create_visualizations(output_path)
        
        print(f"Complete analysis finished. Results saved to: {output_path}")
        
        return results
    
    def run(self, zmap_folder_path: str, 
            sample_sizes_csv_path: str,
            output_directory: Union[str, Path],
            contrast: str = "GROUP",
            mask_path: Optional[str] = None,
            save_all: bool = True,
            debug: bool = False) -> Dict:
        """
        Run the complete workflow including data extraction and meta-analysis.
        
        Parameters:
        -----------
        zmap_folder_path : str
            Path to folder containing Zmap.nii.gz files
        sample_sizes_csv_path : str
            Path to CSV file containing sample sizes
        output_directory : str or Path
            Directory path where all results will be saved
        contrast : str, default "GROUP"
            Contrast of interest to filter for
        mask_path : str, optional
            Path to brain mask file
        save_all : bool, default True
            Whether to save all results to disk
        debug : bool, default False
            Whether to print debugging information
        
        Returns:
        --------
        dict : Dictionary containing all analysis results
        """
        print("="*60)
        print("RUNNING COMPLETE NIFTI META-ANALYSIS WORKFLOW")
        print("="*60)
        
        # Convert output_directory to Path object
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Find Z-maps
        print("Step 1: Finding Z-maps...")
        df = extract_zmap_data(zmap_folder_path, debug=debug)
        if df.empty:
            raise ValueError(f"No Z-maps found in {zmap_folder_path}")
        print(f"  Found {len(df)} Z-map files")
        
        # Step 2: Extract sample sizes
        print("Step 2: Extracting sample sizes...")
        df_samples = extract_sample_sizes(sample_sizes_csv_path)
        if df_samples.empty:
            raise ValueError(f"No sample sizes extracted from {sample_sizes_csv_path}")
        print(f"  Found sample sizes for {len(df_samples)} sites")
        
        # Step 3: Filter for effect of interest
        print(f"Step 3: Filtering for contrast '{contrast}'...")
        filtered_df = df[df['contrast'] == contrast]
        if filtered_df.empty:
            available_contrasts = df['contrast'].unique().tolist()
            raise ValueError(f"No data found for contrast '{contrast}'. Available contrasts: {available_contrasts}")
        print(f"  Found {len(filtered_df)} files for contrast '{contrast}'")
        
        # Step 4: Merge with sample sizes
        print("Step 4: Merging with sample sizes...")
        merged_df = pd.merge(filtered_df, df_samples, on='site', how='inner')
        if merged_df.empty:
            raise ValueError("No matching sites found between Z-maps and sample sizes")
        print(f"  Successfully matched {len(merged_df)} sites")
        
        # Display summary of data
        print("\nData Summary:")
        print("-" * 40)
        for _, row in merged_df.iterrows():
            print(f"  {row['site']}: n={row['n']}")
        print(f"Total sample size: {merged_df['n'].sum()}")
        
        # Step 5: Prepare data for meta-analysis
        print("\nStep 5: Preparing data for meta-analysis...")
        image_paths = dict(zip(merged_df['site'], merged_df['stat_map']))
        sample_sizes = dict(zip(merged_df['site'], merged_df['n'].astype(int)))
        
        # Step 6: Load images
        print("Step 6: Loading images...")
        self.load_images(image_paths, mask_path, sample_sizes)
        
        # Step 7: Run complete analysis
        print("Step 7: Running meta-analysis...")
        results = self.run_complete_analysis(output_directory, save_all=save_all)
        
        # Final summary
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Sites analyzed: {self.n_sites}")
        print(f"Total sample size: {sum(self.sample_sizes.values())}")
        print(f"Brain voxels analyzed: {np.sum(self.mask)}")
        print(f"Results saved to: {output_directory}")
        print("="*60)
        
        return results
    
    def workflow(self, report_dir, result_dir, model_name, model_formula, model_Subjects, mask1):
        """
        Workflow of the complete meta-analysis using site-specific statistical outputs.
        
        Args:
            report_dir (str):   Path to the folder of the target data type in the Reports folder.
            result_dir (str):   Path to the folder of statistical outputs of the target data type.
            model_name (str):   Name of the model, e.g., 'model_01'.
            model_formula(str): Texts of the mega-model, e.g., 'lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))' that is changed to a site-specific meta-model, i.e., 'lm(Yvar ~ GROUP + AGE + SEX)'.
            model_Subjects(str):Path to the datatype- and model-specific Subjects's .csv file.
            mask1 (str):        Path to the inclusive mask image, e.g., '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/brain_mask.nii'.
        
        Outputs:
            save meta-analysis results into the same folder.
        """ 
        
        # Print information
        t0 = time.time() # start time
        print(f"\nMeta-analysis ...")
        
        # Define paths
        zmap_folder = os.path.join(result_dir,"Meta") # e.g., "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_16/Results/reHo2_reho/Meta"
        csv_file = model_Subjects.replace(".csv", "_Table_S1.csv")# e.g., "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_16/Reports/reHo2_reho/Subjects/M01_Table_S1.csv"
        mask_file = mask1 #e.g., "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/tpl-MNI152NLin2009cAsym_res-02_label-GM_binary_mask_80percent.nii.gz"
        
        # Effects of interest
        # Get all immediate subfolder names below "statistic" folders
        subfolder_names = [
            subfolder.name 
            for statistic_folder in Path(zmap_folder).rglob("statistic") 
            if statistic_folder.is_dir()
            for subfolder in statistic_folder.iterdir() 
            if subfolder.is_dir()
        ]
        # Get unique values
        unique_subfolder_names = list(set(subfolder_names))
        
        # Loop through effects of interest
        for effect in unique_subfolder_names:
            print(f"\n  Effect of interest: {effect}")
            # Output directory       
            output_dir = os.path.join(report_dir, "Meta", effect, model_name)
            
            # Run complete workflow
            results = self.run(
                zmap_folder_path=zmap_folder,
                sample_sizes_csv_path=csv_file,
                output_directory=output_dir,
                contrast=effect,
                mask_path=mask_file,
            )
 
        # print ending info
        print(f"\nMeta-analysis completed! Model_name = {model_name}\nTime elapsed (in secs): {time.time()-t0}\n")


# # Example usage
# if __name__ == "__main__":
    
#     # Define paths
#     zmap_folder = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_16/Results/reHo2_reho/Meta"
#     csv_file = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_16/Reports/reHo2_reho/Subjects/M01_Table_S1.csv"
#     mask_file = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/tpl-MNI152NLin2009cAsym_res-02_label-GM_binary_mask_80percent.nii.gz"
#     output_directory = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_16/test/meta_analysis_results"
    
#     # Run complete workflow with one method call
#     results = Meta().run(
#         zmap_folder_path=zmap_folder,
#         sample_sizes_csv_path=csv_file,
#         output_directory=output_directory,
#         contrast="GROUP",
#         mask_path=mask_file,
#     )