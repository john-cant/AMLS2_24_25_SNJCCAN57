# **AMLS II Assessment:Single Image Super Resolution Challenge**
This details the design, development and testing of a machine learning (ML) approach for single-image super-resolution, focusing on the NTIRE 2017 challenge [1] tracks. Utilizing the DIV2K dataset for training, I initially implemented a SRResNet model [7] based on Deep Neural Network (DNN) architectures to upscale low-resolution im-ages to high-resolution counterparts. 
The methodology encompasses data loading and pre-processing, and a three-part approach of base model, hyperparameter selection and tuned model rigorous training, optimization and testing. Performance was assessed using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) metrics on a validation subset of DIV2K. Results demonstrate the efficacy of the proposed model in enhancing image resolution, demonstrating improvements in image restoration and enhancement. 
Building off this initial SRResNet model, I also explored other DIV2K datasets and alternative models such as EDSR and drew additional conclusions.

The key scripts to run the various hyperparameter selections, training and testing were implemented in Jupyter notebooks running on a specific Anaconda configured environment. These reference both external and locally developed Python library - AMLS_common.py – the later to reduce the need for duplicated code and to improve the relevance, efficiency and readability of the scripts.

## **Installation Instructions:**
This project is intended to be easy to run and need the minimum of installation. The files can be run from a copy of the GitHub structure.
All the Python packages required are listed in the Packages Required section below. There is a single main.py file that will run the main scripts.
In addition a copy of a local code library, - developed for these specific task - called AMLS_common.py needs to be included in the base folder.
Each folder should have a metrics folder which will be used to store results files of various types. If this does not exist then main will create it.


## **Usage Examples:**
python main.py 
Will run the main Tasks Jupyter Notebook files 


## **Features:**
There are several main scripts/Jupyter notebooks:
- Task_Base: runs the base SRResNet model
- Task_Multi: designed for hyperparameter selection
- Task_Tune: runs a single model with select hyperparameters and produces output using Validation datasets
- Task_Tune_2: takes the optimised HyperParameters and runs the model against the Track 2 Dataset, outputting results

## **Files Output**
The scripts will output a number of files to store results and allow easy passing of parameters between scripts. These are:

### Hyper Outputs
- Run file summary of all the results and values for the hyperparameter sets generated and trained in Hyper
- Parameter file containing the _best case_ hyperparameter set in format thta can be read in by Tune

### Tune Outputs (both have same timestamp)
- Metrics spreadsheet containing an epoch by epoch list of outputs from history
- Summary text file containing the model summary

## **Packages Required:**
- io
- os
- time
- datetime         
- numpy
- glob
- sys
- matplotlib.pyplot
- tensorflow
- tensorflow.keras.models 
- tensorflow.keras.layers
- tensorflow.keras.optimizers
- tensorflow.keras.losses
- dataclasses
- pandas
- tqdm
- sklearn.linear_model
- sklearn.model_selection
- sklearn.metrics
- sklearn.ensemble
- sklearn.svm
- skimage.metrics
- google.colab
- nbconvert

## **Function details within AMLS Common**
A core design principle is to keep as much of the com-mon functionality in this single Python library to reduce length, duplication, and improve efficiency and readability of all model scripts. The following describes the core components in this library.

_Data Classes_
-	HyperParameter: data class to allow storage and passing of set of hyperparameters as a structure. Contains possible defaults, and also specific list, load and save Excel subfunctions
-	RunResult: to allow storage and passing of summary model run results

_Data Load_
There are two datasets used, one per Task. 

_Models_
-	SRResNet Base: This is the base model used to generate initial results using the Base scripts. It is fully functional but has a number of hyperparameters defaulted so is less flexible.
-	SRResNet Tune: This is an improvement of the base model with a full set of hyperparameters that can be flexed although it still retains the same base architecture.
-	SRResNet Plus: Building on SRResNet Tune, this introduces an enhanced architecture with added RRDB functionality.
-	EDSR: This is an implementation of simplified EDSR architecture, with ability to flex hyperparameters.
-	SRResNet Tune 2: Based on the tune model above but with changes for Track 2 data image sizes.

_Utility Functions_
-	Dataset to NumPy: converts a Tensorflow dataset into a NumPy arrays
-	Get TimeStamp: Gets date/timestamp to allow traceable filenames for run metrics/parameter files
-	TDQM Epoch Progress: to allow displaying of progress metric to track longer running epochs
-	StopOverfittingCallback: implements overfitting reduction function as defined in 4.1.4

_Analysis Functions_
These support rich analysis plus graphing and results storage in structured files.
-	Graph and Save: Calls Graph and History to Excel
-	Graph: outputs multi-quadrant graph for accuracy and loss for given run history. The skip parameter can define the first epoch to be plotted. This is useful where much of the change occurs in the first few epochs suppressing later differences due to scaling.
-	Graph Compare: allows display of two model runs from metrics files on a single plot.
-	History to Excel: flexibly writes a run history set of accuracy and loss values to an excel file and a linked text file with model structure and parame-ters.
-	Hyper Process: flexibly reads run history and writes to run result dataframe to simplify analysis.
-	Analyse Run: saves hyper run output and then se-lects best set using combination of criteria.
-	Analyse HyperParameters: uses several techniques, including linear and random forest regression to gauge their impact.
