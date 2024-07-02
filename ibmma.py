import os  # for file operations
import pandas as pd # for working with csv files
import multiprocessing as mp # for parallel processing
from SDL_functions.startup import Clean
from SDL_functions.mega_analysis import Mega

class IBMMA:
    def __init__(self, file_path_para = 'path_para.xlsx', new_subjects=False, num_processes=None):
        self.file_path_para = file_path_para # path to the 'path_para.xlsx' file for information of paths & parameters
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count() # use all CPUs if not specified
        self.Clean = Clean(self.file_path_para) # self.clean.df_all_sheets contains all sheets of path_para.xlsx
        self.Subjects = os.path.join('Processes', 'Subjects.csv') # path to Subjects.csv
        self.df_subjects = [self.Clean.match_files() if (not os.path.isfile(self.Subjects)) or new_subjects else pd.read_csv(self.Subjects)] # Load existed Subjects.csv OR make it if it does not exist or new_subjects==True
        
    def pipeline(self):
        """
        The pipeline to run steps one-by-one.
        """
        # Loop across data patterns
        for _, data_pattern in self.Clean.df_all_sheets['data_pattern'].iterrows():
            process_dir  = os.path.join('Processes', data_pattern['NAME'])
            result_dir   = os.path.join('Results', data_pattern['NAME'])
            num_segments = 50 # default = 50 segments
            
            Mega().mask(self.Subjects, 'FULL_PATH_DATA_'+data_pattern['NAME'], 'FULL_PATH_MASK_'+data_pattern['NAME'], process_dir)
            Mega().flatten(self.Subjects, 'FULL_PATH_DATA1_'+data_pattern['NAME'], process_dir)
            Mega().segment(process_dir, num_segments) 
            
            # Loop across statistical models
            for _, model in self.Clean.df_all_sheets['models'].iterrows(): 
                model_name    = model['NAME']
                model_formula = model['FORMULA']
                path_R_stat   = os.path.join('SDL_functions','R_modelling_parallel.R') # path to the R script for statistical analysis
                
                Mega().stat(process_dir, os.path.join(process_dir, 'stats'), path_R_stat, self.Subjects, model_name, model_formula)
                Mega().concatenate(process_dir, result_dir, model_name)
                Mega().reverse(process_dir, result_dir, model_name)
                # Mega().p_correct(result_dir, model_name)

if __name__ == "__main__": 
    
    ibmma = IBMMA(new_subjects=False)
    ibmma.pipeline()   
    print("\n\n Finished !!!\n\n")