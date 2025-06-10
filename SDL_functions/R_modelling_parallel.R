# This R script runs mass-univariate analysis by parallely fitting
# a given model (in text strings) on multiple outcome variables (Ys) based on the same predictors (Xs), 
# and save the statistical outputs (TIDY) as well as model estimations (GLANCE) following Broom style.

# This script could be run using terminal command with 5 arguments such as
# Rscript path/to/R_modelling_parallel.R arg1 arg2 arg3 arg4 arg5
# --- arg1, full path to xfile.csv, which contains predictors (row = observation, column = variable).
# --- arg2, full path to yfile.csv, which contains outcome variables (row = observation, column = variable).
# --- arg3, full path to the output directory.
# --- arg4, model name, e.g. "model_01"
# --- arg5, texts of model formula, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))" or "lm(Yvar ~ GROUP + AGE + SEX)".
# --- arg6, variable name of the random factor for meta analysis, e.g. "SITE", so that simplified model is running per level (per site). default = "Mega" for running mega-analysis
# An example of terminal command:
# Rscript ./SDL_functions/R_modelling_parallel.R ./Progress/Subjects.csv ./Progress/atlas_conn_SFC/static_fc/segmented/V47000_48880.csv  ./Progress/atlas_conn_SFC/static_fc/stats/V47000_48880/Mega "model_01" "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))"

# Copyright (c) 2024 Delin Sun (ds366@duke.edu; sundelinustc@gmail.com)
# All rights reserved.
# This script is the property of Delin Sun.
# No part of this script may be reproduced in any form without the prior permission of Delin Sun.

# for test purpose only
xfile   <- '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_13/Reports/reHo_reho/Subjects/M01_Seth.csv'
yfile   <- '/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/IBMMA-v0.1.1-beta_13/Processes/reHo_reho/segmented/V476080_497720.csv'
out_dir <- 'Processes/reHo_reho/stats/V476080_497720/Mega'
# out_dir <- 'Processes/reHo_reho/stats/V476080_497720/Meta'
model_name <- 'M01_Seth'
# formula <- "lm(Yvar ~ GROUP * AGE + SEX)"
formula <- 'lmer(Yvar ~ GROUP_Seth_PTSD_TEC + AGE + AGE2 + SEX + (1|SITE))'
# formula <- "lmer(Yvar ~ 1 + (1|SITE))"
meta_factor <- "Mega"

# (0) get command line arguments
args <- commandArgs(trailingOnly = TRUE)
xfile   <- args[1]
yfile   <- args[2]
out_dir <- args[3]
model_name <- args[4]
formula <- args[5]
meta_factor <- args[6]

# (1) Packages
# install pacman to load the other packages
if (!require(pacman)){install.packages('pacman', dependencies = TRUE)}
library(pacman)
# basic packages
packages <- c('data.table','foreach','doParallel','broom')
# specific packages for the given model formula
if (grepl("lmer\\(", formula)){packages <- c(packages,'lme4','lmerTest','broom.mixed')}
# load all packages of interest
do.call(p_load, as.list(packages))

# (2) Helper functions
# (2.1) function to fit the model and return the tidy and glance results
fit_model <- function(yvar, formula, dt, nobs_extra, random_var=NULL) {
  # input
  # --- yvar, a column name in dt, e.g., "V123"
  # --- formula, a string of model, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))", 
  #              or "lm(Yvar ~ GROUP + AGE + SEX)"
  # --- dt, data.table of independent (Xs) and dependent (Ys) variables
  # --- nobs_extra, data.table of categorical variables, their levels, and sample size per level.
  # --- random_var, column name(s) of random factors
  # output (return a list of tidy & glance)
  # --- tidy, a data table of the model's statistical outputs
  # --- glance, a data table of model summaries

  # drop missing values & empty string
  # dt <- dt[complete.cases(dt) & !Reduce(`|`, lapply(dt, function(x) trimws(x) == "" | x == "NA" | x == "nan" | x == "NaN" | x == "None" | x == "null"))]
  dt <- na.omit(dt)

  # modify texts of the given formula
  # replace "Yvar" with yvar, which is a column name, e.g., 'V123'
  str_formula <- gsub("Yvar", yvar, formula)
  # replace the last element of the text with new ending texts
  str_formula <- sub(".$", ", data=dt)", str_formula)

  # Fit the model using the specified model function
  fit <- eval(parse(text = str_formula)) # any kind of formula, love it so much!

  # Get the tidy and glance results as data.tables
  tidy <- as.data.table(tidy(fit))
  glance <- as.data.table(glance(fit))

  # Inner Function to find columns of categorical variables (<= 4 unique values) in xdt
  inner_count_categorical_levels <- function(xdt, random_factors = NULL, max_unique_values = 4) {
    # Identify categorical columns, excluding random factors
    is_categorical <- function(x) {length(unique(x)) <= max_unique_values}
    categorical_cols <- names(xdt)[sapply(xdt, is_categorical)]
    
    # Exclude random factors if provided
    if (!is.null(random_factors)) {categorical_cols <- setdiff(categorical_cols, random_factors)}
    
    # Calculate level counts for categorical variables
    results <- data.table()
    for (col in categorical_cols) {
      level_counts <- xdt[, .N, by = col]
      level_counts[, Variable := col]
      setnames(level_counts, col, "Level")
      level_counts[, Name := paste("nobs", Variable, Level, sep = "_")]
      setcolorder(level_counts, c("Variable", "Level", "Name", "N"))
      results <- rbindlist(list(results, level_counts), use.names = TRUE)
    }
    
    return(results)
  }

  # supplement glance results with extra nobs of different levels of categorical variable(s)
  nobs_extra1 <- inner_count_categorical_levels(dt, random_var) # number of unique values <=4 is categorical variable
  nobs_extra1[, Level := as.character(Level)] # convert the column "Level" from num to character

  # merge nobs_extra & nobs_extra1
  nobs_extra <- nobs_extra[nobs_extra1, on = c("Variable", "Level", "Name"), N := i.N]
  nobs_extra[, N := nafill(N, fill = 0)] # NA with 0 in the specific column "N"

  # Check if nobs_extra is not empty
  if (nrow(nobs_extra) > 0) {
    # Convert nobs columns to list format for assignment
    nobs_list <- as.list(nobs_extra[, N])
    names(nobs_list) <- nobs_extra[, Name]
    
    # Add new columns to glance_results
    glance[1, names(nobs_list) := nobs_list]
  }

  # Return a list of tidy and glance
  return(list(tidy = tidy, glance = glance))
}

# (2.2) function to fit models across multiple dependent variables parallely
fit_model_parallel <- function(xdt, ydt, formula, fit_model, nobs_extra, random_var=NULL){
  # input
  # --- xdt, data.table of independent variables (Xs).
  # --- ydt, data.table of dependent variables (Ys).
  # --- formula, a string of model, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))", 
  #              or "lm(Yvar ~ GROUP + AGE + SEX)".
  # --- fit_model, the pre-defined function fit_model.
  # --- nobs_extra, data.table of categorical variables, their levels, and sample size per level.
  # --- random_var, column name(s) of random factors

  # create a list where each element is a list containing a data.table of ydt column and the corresponding column name
  ydt_list <- lapply(colnames(ydt), function(y) list(data = data.table(ydt[[y]]), name = y))

  # packages for parallel processing
  if (grepl("lmer\\(", formula)) {
    # for linear mixed effects model
    packages <- c("data.table", "lme4", "lmerTest", "broom.mixed")
  } else {
    packages <- c("data.table", "broom")
  }

  # Inner function to combine parallel processing outputs
  comb <- function(x, ...) {
    # input
    # --- series of dataframes generated by parallel processing
    # output
    # --- a big dataframe by vertically combing all dataframes
    lapply(seq_along(x), function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
    }

  # the number of cores that could be used
  # cores <- min(detectCores(), dim(ydt)[2]) # minimum of possible cores and number of outcome variables
  cores <- detectCores() - 1
  # use all or some of them
  cl <- makeCluster(cores) # default: use all of the cores to speed up
  # register the parallel backend
  registerDoParallel(cl)

  # initialize two lists to store the tidy and glance results
  tidy_results <- list()
  glance_results <- list()

  # run the fit_model function in parallel for each dependent variable & combine the results in a list
  results <- foreach(ydt1 = ydt_list, .packages = packages, .combine='comb', .multicombine=T, .init=list(list(), list())) %dopar% {
      tryCatch({
        # # for test purpose only
        # ydt1 <- ydt_list[[3372]]

        # merge the two data.tables
        dt <- cbind(xdt, ydt1$data) # not using merge by fID to save time
        # get yvar name
        yvar <- ydt1$name
        # model fit
        res <- fit_model('V1', formula, dt, nobs_extra, random_var) # the column name of ydt1$data is always "V1"
        # tidy results
        tidy_results <- res$tidy
        tidy_results$Yvar <- yvar # New column of Yvar to be added
        # glance results
        glance_results <- res$glance
        glance_results$Yvar <- yvar # New column of Yvar to be added

      # return for each Y
      list(tidy_results, glance_results)
      },
      error = function(e) {
        # Log the error
        cat("Error happens in processing", ydt1$name, ":", conditionMessage(e), "\n")
      
        # Return NULL for errors, which will be filtered out later
        NULL
      })
    }

  # stop cluster
  stopCluster(cl)

  # list to dataframe of outputs
  TIDY   <-  do.call(rbind, results[[1]]) # statistical outputs (coef., t-value, p-value, etc.)
  GLANCE <-  do.call(rbind, results[[2]]) # model estimations

  # return
  return(list(TIDY = TIDY, GLANCE = GLANCE))
}

# (2.3) function to save broom TIDY results into csv files
save_broom_parallel <- function(TIDY, GLANCE, outdir, model_name){
  # input
  # --- TIDY, a data table of statistical outputs including statistics & p values
  # --- GLANCE, a data table of model estimations
  # --- outdir, the directory for storing all temporal results
  # --- model_name, model name, e.g., 'model_01'
  # output
  # --- a new folder "TIDY" under outdir, and contains
  #            subcolders corresponding to TIDY columns (e.g., statistic, p.value), and each subfolder contains
  #            subfolders corresponding to TIDY$term (e.g., GROUP, AGE, SEX, except for "Intercept"), and each subfolder contains
  #            csv file (share filename with yfile) of TIDY for the given column (e.g., "p.value") and given term (e.g., "GROUP")
  # --- a new folder "GLANCE" under outdir, and contains
  #            subfolder corresponding to GLANCE columns (e.g., AIC, BIC), and each subfolder contains
  #            csv file (share filename with yfile) of GLANCE for the given column (e.g., "AIC")

  # prnt info
  print("Modelling completed! Now saving results ...")

  # Ensure TIDY and GLANCE are data.tables
  if (!is.data.table(TIDY)) TIDY <- as.data.table(TIDY)
  if (!is.data.table(GLANCE)) GLANCE <- as.data.table(GLANCE)

  # TIDY terms excluding "(Intercept)"
  TIDY_terms <- setdiff(unique(TIDY$term), "(Intercept)")

  # TIDY columns excluding "term" and "Yvar"
  TIDY_cols <- setdiff(colnames(TIDY), c("effect","group","term","Yvar"))
  
  # GLANCE columns excluding "term" and "Yvar"
  GLANCE_cols <- setdiff(colnames(GLANCE), c("Yvar"))

  # combinations of terms & cols
  TIDY_term_cols   <- expand.grid("TIDY",  TIDY_terms, TIDY_cols)
  GLANCE_term_cols <- expand.grid("GLANCE",        "", GLANCE_cols)
  term_cols <- rbind(TIDY_term_cols, GLANCE_term_cols)

  # Inner function to split and save data
  save_data <- function(term_col) {
    # data of interest
    TIDY_GLANCE <- as.character(term_col[1][1,1]) # TIDY or GLANCE
    term1       <- as.character(term_col[2][1,1]) # term of interest, e.g., "GROUP"
    col         <- as.character(term_col[3][1,1]) # column of interest, e.g., "p.value"

    if (TIDY_GLANCE == "TIDY"){
      # for TIDY data table
      dt1 <- TIDY[term == term1, c('Yvar',col), with=FALSE]
      # dt1 <- TIDY[term == term1, list(Yvar, get(col))]
    } else {
      # for GLANCE data table
      dt1 <- GLANCE[,            c('Yvar', col), with=FALSE]
      # dt1 <- GLANCE[, list(Yvar, get(col))]
    }

    # make directory & save into csv files
    term1 <- gsub(":", "..", term1) # change ":" into ".." to avoid problems in naming folders
    dir.create( file.path(outdir, TIDY_GLANCE, col, term1), recursive = TRUE, showWarning = FALSE)
    fwrite(dt1, file.path(outdir, TIDY_GLANCE, col, term1, paste0(model_name,'.csv')))
  }

  # packages for parallel processing
  packages <- c("data.table")

  # Use parallel processing to speed up
  # num_cores <- min(detectCores(), dim(term_cols)[1]) # minimum number of cores to save resources
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  # register the parallel backend
  registerDoParallel(cl)
  foreach(i = 1:dim(term_cols)[1], .packages = packages) %dopar% {save_data(term_cols[i,]) }

  # parLapply(cl, split(term_cols, seq(nrow(term_cols))), save_data)
  stopCluster(cl)

  # end info
  cat('TIDY & GLANCE saved for',model_name,'\n')
}

# (3) main function to fit model and save broom style outputs
parallel_model <- function(xfile, yfile, out_dir, model_name, formula, fit_model_parallel, save_broom_parallel) {
  # arguments
  # --- xfile, fullpath to the csv file of independent variables (row = observation, column = variable)
  # --- yfile, fullpath to the csv file of depedent variables (row = observation, column = variable)
  #            !!! xfile & yfile MUST be matched by rows (i.e., each row represents the info of the same observation)
  # --- out_dir, path to the folder for saving outputs
  # --- model_name, text string of model name, e.g., 'model_01'
  # --- formula, text string of model formula, e.g., "lmer(Yvar ~ GROUP + AGE + SEX + (1|SITE))", or "lm(Yvar ~ GROUP + AGE + SEX)"
  # --- fit_model_parallel, the fit_model_parallel function defiend beforehand
  # --- save_broom_parallel, the save_broom_parallel function defined beforehand
  # output
  # --- a TIDY folder of statistical outputs per column (e.g., estimate, statistic, p.value) and per term (e.g., GROUP, SEX, and AGe, except for (Intercept))
  # --- a GLANCE folder of model estimation per column (e.g., nob, AIC, BIC)

  # output directory (same name as the yfile, just without file extension)
  outdir <- out_dir

  # Define a custom NA strings vector that covers common missing value representations
  na_strings <- c(NA, "NA", "na", "N/A", "n/a", "NULL", "null", "Null",
    "NONE", "none", "None", "NAN", "NaN", "nan", "")

  # read independent variables (Xs)
  xdt <- fread(xfile, na.strings = na_strings)  # Keep strings as character

  # Optional: Add additional check for columns that might still have variants of missing values
  # This helps catch any unusual representations that weren't in the na_strings vector
  for(col in names(xdt)) {
    if(is.character(xdt[[col]])) {
      # Convert additional patterns to NA using regex
      set(xdt, i = which(grepl("^[[:space:]]*$|^[Nn][Aa][Nn]?$|^[Nn]/[Aa]$|^[Nn][Uu][Ll][Ll]$|^[Nn][Oo][Nn][Ee]$", xdt[[col]])), 
          j = col, 
          value = NA)
    }
  }

  # Inner function to subset data.table keeping only variables in formula plus fID
  # and also return the column names of random factors
  subset_dt_by_formula <- function(dt, formula_str) {
    # Extract all variables and add fID
    formula_clean <- gsub("\\s+", "", formula_str)
    vars <- unique(c(unlist(regmatches(formula_clean, gregexpr("[[:alnum:]_\\.]+", formula_clean))), "fID"))
    
    # Extract random factors (variables after | symbol)
    random_pattern <- "\\|([[:alnum:]_\\.]+)"
    random_factors <- unique(unlist(regmatches(formula_clean, gregexpr(random_pattern, formula_clean), invert = FALSE)))
    random_factors <- gsub("\\|", "", random_factors)
    
    # Remove function names and keep only variables in the data.table
    vars_to_keep <- vars[!grepl("^(lm|lmer|glm|glmer)$", vars) & vars %in% names(dt)]
    
    # Return subsetted data.table and random factors
    return(list(
      data = dt[, ..vars_to_keep],
      random_factors = random_factors[random_factors %in% names(dt)]
    ))
  }
  # Keep xdt's columns that appear in the formula, + fID
  result <- subset_dt_by_formula(xdt,formula)
  xdt <- result$data 
  xdt <- na.omit(xdt) # remove all rows with missing value
  cols_xdt <- names(xdt) # column names of xdt
  random_factors <- result$random_factors

  # Inner function to find columns of categorical variables (<= 4 unique values) in xdt
  count_categorical_levels <- function(xdt, random_factors = NULL, max_unique_values = 4) {
    # Identify categorical columns, excluding random factors
    is_categorical <- function(x) {length(unique(x)) <= max_unique_values}
    categorical_cols <- names(xdt)[sapply(xdt, is_categorical)]
    
    # Exclude random factors if provided
    if (!is.null(random_factors)) {categorical_cols <- setdiff(categorical_cols, random_factors)}
    
    # Calculate level counts for categorical variables
    results <- data.table()
    for (col in categorical_cols) {
      level_counts <- xdt[, .N, by = col]
      level_counts[, Variable := col]
      setnames(level_counts, col, "Level")
      level_counts[, Name := paste("nobs", Variable, Level, sep = "_")]
      setcolorder(level_counts, c("Variable", "Level", "Name", "N"))
      results <- rbindlist(list(results, level_counts), use.names = TRUE)
    }
    
    return(results)
  }

  # supplement glance results with extra nobs of different levels of categorical variable(s)
  nobs_extra <- count_categorical_levels(xdt, random_factors) # number of unique values <=4 is categorical variable
  nobs_extra <- na.omit(nobs_extra) # remove levels of NA
  nobs_extra <- nobs_extra[,c("Variable", "Level", "Name")] # no need to keep column of N 
  nobs_extra[, Level := as.character(Level)] # convert the column "Level" from num to character

  # read dependent variables (Ys)
  ydt <- fread(yfile, header=TRUE, na.strings = na_strings)
  cols_ydt <- names(ydt)

  # Merge xdt and ydt through their common values in fID
  dt_merged <- merge(xdt, ydt, by = "fID", all = FALSE)

  # Update xdt & ydt
  xdt <- dt_merged[,..cols_xdt]
  ydt <- dt_merged[,..cols_ydt]
  ydt <- ydt[, !c("fID")] # Drop fID in ydt

  # Inner function to check if a column is "bad" - meaning it is:
  # - constant (all the same value)
  # - full of NA
  # - full of NaN
  # - full of empty entries
  # - has only one unique non-NA/non-NaN value
  # - has 5 or fewer numeric values
  is_bad_column <- function(dt, col_name) {
    col <- dt[[col_name]]
    if (all(is.na(col))) return(TRUE) # Fast checks using data.table optimizations
    non_na_values <- col[!is.na(col)] # Count non-NA values using data.table's efficient operations
    if (length(non_na_values) > 0 && all(non_na_values == "")) return(TRUE) # Check if all non-NA values are empty strings
    if (length(non_na_values) > 0 && uniqueN(non_na_values) == 1) return(TRUE) # Check if all non-NA values are the same (constant column)
    numeric_count <- sum(!is.na(col) & is.numeric(col)) # Count numeric values efficiently
    if (numeric_count <= 5) return(TRUE)
    return(FALSE)
  }

  # Identify bad columns
  bad_columns <- names(ydt)[sapply(names(ydt), function(col) is_bad_column(ydt, col))]

  # model fitting & save outputs only when not all outcome variables are constant, NAs, NaNs, or empty entries
  if (length(bad_columns) < length(colnames(ydt))){
    # remove the bad columns (if any) from ydt
    if (length(bad_columns) > 0){ydt[, (bad_columns) := NULL]}

      # if meta_factor is one of demographic/clinical variables, and is a random factor in the formula
      if ((meta_factor %in% names(xdt)) & (meta_factor==strsplit(gsub("\\(|\\)", "", formula), "\\|")[[1]][2])){# Meta-analysis
        # Filter subjects per unique value of meta_factor (SITE)
        for (val in unique(xdt[[meta_factor]])) {
            indices <- which(xdt[[meta_factor]] == val)# Indices of a particular meta_facotr (e.g. SITE), e.g. Duke
            xdt1 <- xdt[indices]
            ydt1 <- ydt[indices]

            # Revise formula by removing specifid random factor & change lmer to lm
            # e.g., "lmer(Yvar ~ GROUP * AGE + AGE2 + SEX + (1|SITE))"
            # to    "lm(Yvar ~ GROUP * AGE + AGE2 + SEX)"
            formula1 <- gsub("lmer\\(", "lm(", paste0(strsplit(formula, paste0(" \\+ \\(1\\|", meta_factor, "\\)"))[[1]][1], ")"))
            
            # Revise formula1 by removing variables that contain constant values in xdt1
            # e.g., "lmer(Yvar ~ GROUP * AGE + AGE2)" if SEX is constant
            for (var in names(xdt1)) {
                if (length(unique(xdt1[[var]])) == 1) {
                    formula1 <- gsub(paste0("\\+\\s*", var), "", formula1)
                    formula1 <- gsub(paste0("\\s*\\+\\s*", var), "", formula1)
                }
            }

            # fit model parallely
            output <- fit_model_parallel(xdt1, ydt1, formula1, fit_model, nobs_extra, random_var=random_factors)
            # save TIDY & GLANCE parallely
            save_broom_parallel(output$TIDY, output$GLANCE, file.path(outdir,val), model_name)
        }
        
      } else {# Mega-analysis
          # fit model parallely
          output <- fit_model_parallel(xdt, ydt, formula, fit_model, nobs_extra, random_var=random_factors)
          # save TIDY & GLANCE parallely
          save_broom_parallel(output$TIDY, output$GLANCE, outdir, model_name) # site-specific statistical outputs
      }

  }
}

# (4) run the main function based on terminal command
parallel_model(xfile, yfile, out_dir, model_name, formula, fit_model_parallel, save_broom_parallel)