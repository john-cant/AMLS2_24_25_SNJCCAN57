""" AMLS COMMON FUNCTIONS
"""
# pylint: disable=no-member
## Common code to support all tasks of AMLS Assigmnent
## including the loading the MedMNIST data files into tensorflow format
## loading of hyperparameters into single data class structure
## and handling of NN history results: graphing and storing results in files
## Used across both assignment tasks
## import common libraries
## Revision History
## 11012025 Added compare graph function and overfitting callback rather than previous manual option
## 16032025 Added AMLS2 base functions, plus minor updates to existing functions
## 24032025 Played with lpips_loss and updated data loading functions
## 26032025 Fixed/improved setting of run_result using flexible get_run_metrics function
## 27032025 Integrated srresnet_plus into this file and updated hyperparameters
## 29032025 Fixed activation layer parameters in sressnet_plus
## 29032025 Added edsr_plus with item parameters
## 31032025 Added srresnet_plus_2 to overcome Track2 image size issues
## 01042025 Updated srresnet model naming to have base (base), tune (all hyperparams) and 
##          plus (RRDB) models for more logical progression


#################################################### LIBRARY IMPORTS ##############################
## standard python libraries
import datetime
import glob
import functools
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
## import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Multiply,Add,Layer,Lambda
from tensorflow.keras.layers import Input,Conv2D,Dropout,PReLU, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam,AdamW
##from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
##from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.nn import depth_to_space
## sklearn to allow analysis of hyperparameter choices
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skimage.metrics import structural_similarity as ssim

## Loading the data file using a loader
DATA_FLAG      = 'X2'        ## defines which dataset to load
CROP_SIZE      = 224         ## HR crop size
IMG_SIZE       = 224

#################################################### SET UP DATACLASSES ########################
@dataclass
class HyperParameters:
    """ data class to allow storage and passing set of hyperparameters as structure
    """
    learning_rate: float
    batch_size: int
    num_epochs: int
    optimise: str
    loss: str
    num_filter: int
    strides: int = 1
    padding: str = "same"
    dropout_rate: float = 0.2
    layers: int = 3
    activation: str = "Prelu"
    kernel_size: int = 3
    scale: int = 4
    momentum: float = 0.8
    epsilon: float = 0.00001

    def list_parameters(self):
        """ lists all attributes and values in HyperParameters class
        """
        result = ""   ## initalise result
        ## Loop through attributes and get their values
        for field in fields(HyperParameters):
            attribute_name = field.name
            value = getattr(self, attribute_name)
            result= result+(f"{attribute_name}: {value}"+"\n")
        return result

    @classmethod
    def load_excel(cls, file_path: str):
        """
        Loads a single dataclass instance from an Excel file.
        Returns HyperParameter: An instance of the dataclass with values from the Excel file.
        """
        # Read the Excel file into a DataFrame
        df_load = pd.read_excel(file_path)
        ## Ensure the file contains at least one row
        if df_load.empty:
            raise ValueError(f"The file {file_path} is empty.")
        ## Convert the first row of the DataFrame into a dictionary
        data_dict = df_load.iloc[0].to_dict()
        ## Pass the dictionary as keyword arguments to the dataclass constructor
        return cls(**data_dict)

    def save_excel(self, file_path: str):
        """
        Saves the dataclass instance to an Excel file.
        """
        ## Convert the dataclass to a dictionary
        data = self.__dict__
        ## Convert the dictionary into a pandas DataFrame
        df_save = pd.DataFrame([data])  # Wrap in a list to create a single-row DataFrame
        ## Save the DataFrame to Excel
        df_save.to_excel(file_path, index=False)

@dataclass
class RunResult:
    """ data class to allow storage and passing of summary run results as structure
    """
    min_loss: float = None
    max_acc: float = None
    last_loss: float = None
    last_acc: float = None
    min_val_loss: float = None
    max_val_acc: float = None
    last_val_loss: float = None
    last_val_acc: float = None
    var_loss: float = None
    var_acc: float = None

    def list_runresult(self):
        """ lists all attributes and values in RunResult class
        """
        result = ""   ## initalise result
        ## Loop through attributes and get their values
        for field in fields(RunResult):
            attribute_name = field.name
            value = getattr(self, attribute_name)
            result= result+(f"{attribute_name}: {value}"+"\n")
        return result

def get_run_metrics(df, **metric_operations):
    """ Gets specified metrics from a DataFrame and directly loads them into RunResult.
        the column for data as input to the metric is the first part of the input
        the exact metrics to be stored are specified by the operation part of the input
    """
    results = {}
    for result_key, (column_name, operation) in metric_operations.items():
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found. Setting {result_key} to None.")
            results[result_key] = None ## no column found so set to None
            continue
        try:
            if operation == 'min':
                results[result_key] = float(df[column_name].min())
            elif operation == 'max':
                results[result_key] = float(df[column_name].max())
            elif operation == 'first':
                results[result_key] = float(df[column_name].iloc[0])
            elif operation == 'last':
                results[result_key] = float(df[column_name].iloc[-1])
            elif operation == 'var':
                results[result_key] = float(df[column_name].var())
            else:
                print(f"Warning: Invalid operation '{operation}' for column '{column_name}'\
                      . Setting {result_key} to None.")
                results[result_key] = None ## invalid operation so set to None
        except Exception as e:
            print(f"Error calculating {result_key}: {e}. Setting to None.")
            results[result_key] = None
    return RunResult(**results) ## dict

class TqdmEpochProgress(tf.keras.callbacks.Callback):
    """ simple bar to show progress during model execution
    """
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None

    def on_train_begin(self, logs=None):
        """ set up for start of training run
        """
        self.progress_bar = tqdm(total=self.total_epochs, desc="Epoch Progress", unit="epoch")

    def on_epoch_end(self, _, logs=None):
        """ update after each epoch
        """
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs)

    def on_train_end(self, logs=None):
        """ close out at end of training run
        """
        self.progress_bar.close()

class StopOverfittingCallback(tf.keras.callbacks.Callback):
    """ Callback to restrict overfitting
    """
    def __init__(self, patience=3, threshold=0.1):
        """
        Args:
            patience (int): Number of epochs to allow overfitting before stopping.
            threshold (float): Maximum allowed difference between training and validation loss.
        """
        super(StopOverfittingCallback, self).__init__()
        ## initialise
        self.patience          = patience       ## Number of epochs overfitting
        self.threshold         = threshold      ## What is a material overfit threshold
        self.overfitting_count = 0              ## Count epochs with overfitting

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        ## Check if validation loss is significantly higher than training loss
        if val_loss is not None and train_loss is not None:
            gap = val_loss - train_loss
            if gap > self.threshold:
                self.overfitting_count += 1
                print(f"Overfitting detected at epoch {epoch+1}: Loss Gap = {gap:.4f}")
            else:
                self.overfitting_count = 0  # Reset if no overfitting in this epoch
            ## Stop training if material overfitting persists for 'patience' epochs
            if self.overfitting_count >= self.patience:
                print("Stopping training due to persistent overfitting.")
                self.model.stop_training = True

############################# DATA SET LOAD ##########################

def load_data(lr_train_folder,hr_train_folder,lr_val_folder,hr_val_folder,
              batch_size,upscale_factor):
    """ loading all the images from local copies of the Training and Validation datasets 
        tests applied to confirm there is a complete set and they are a common size
        HR and LR images which when loaded are linked in pairs 
        Each dataset is stored in tensors to support the various ML models
    """
    ## Get sorted lists of training image paths
    lr_train_paths = sorted(tf.io.gfile.glob(lr_train_folder + "/*.png"))
    hr_train_paths = sorted(tf.io.gfile.glob(hr_train_folder + "/*.png"))
    ## print("Training image paths",len(lr_train_paths),len(hr_train_paths))
    assert len(lr_train_paths) == len(hr_train_paths),"Mismatch between LR and HR train paths!"
    ## Create TensorFlow dataset of paths
    lr_dataset = tf.data.Dataset.from_tensor_slices(lr_train_paths)
    hr_dataset = tf.data.Dataset.from_tensor_slices(hr_train_paths)
    ## Zip the datasets together to create (LR, HR) pairs
    train_dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))
    ## Map the function to load images
    load_image_pair_partial = functools.partial(load_image_pair, upscale_factor=upscale_factor)
    train_dataset = train_dataset.map(load_image_pair_partial, num_parallel_calls=tf.data.AUTOTUNE)
    ## Batch and shuffle the dataset
    train_dataset = train_dataset.batch(batch_size).shuffle(100).prefetch(tf.data.AUTOTUNE)
    ## Get sorted lists of validation image paths
    lr_val_paths = sorted(tf.io.gfile.glob(lr_val_folder + "/*.png"))
    hr_val_paths = sorted(tf.io.gfile.glob(hr_val_folder + "/*.png"))
    ## print("Validation image paths",len(lr_val_paths),len(hr_val_paths))
    assert len(lr_val_paths) == len(hr_val_paths), "Mismatch between LR and HR validation paths!"
    ## Create TensorFlow dataset of paths
    lr_dataset = tf.data.Dataset.from_tensor_slices(lr_val_paths)
    hr_dataset = tf.data.Dataset.from_tensor_slices(hr_val_paths)
    ## Zip the datasets together to create (LR, HR) pairs
    val_dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))
    ## Map the function to load images
    val_dataset = val_dataset.map(load_image_pair_partial, num_parallel_calls=tf.data.AUTOTUNE)
    ## Batch and shuffle the dataset
    val_dataset = val_dataset.batch(batch_size).shuffle(100).prefetch(tf.data.AUTOTUNE)
    verbose = 1
    if verbose == 1:
        ## print summary stats for training dataset
        print("\nSummary metrics for train_dataset")
        print("type:",type(train_dataset))
        print("length:",len(train_dataset))
        print("shape:",train_dataset)
    return train_dataset,val_dataset

def test_model(val_dataset,model,batch_size,filebase):
    """ test model
    """
    tag       = 0
    psnr_list = []
    ssim_list = []
    ## Take min of batch_size or 4 images from validation dataset
    if batch_size < 10:
        testing_set = 10
    else:
        testing_set = batch_size
    for lowres, highres in val_dataset.take(testing_set):
        ## Extract first image from batch
        lowres  = lowres[0].numpy()  ## Convert Tensor to NumPy array
        highres = highres[0].numpy()
        ## Resize instead of cropping
        lowres = tf.image.resize(lowres, (224, 224)).numpy()
        ## Expand dims to match model input shape
        lowres_input = np.expand_dims(lowres, axis=0)  ## Shape: (1, 224, 224, 3)
        ## Get predictions
        preds = model.predict(lowres_input)  ## Model outputs batch shape (1, H, W, C)
        preds = preds[0]  ## Remove batch dimension
        ## Plot results
        print("low & pred")
        plot_results(lowres, preds)
        print("high & pred")
        plot_results(highres, preds)
        psnr_list.append(calculate_psnr(highres,preds))
        print("psnr",calculate_psnr(highres,preds))
        ## Find the smallest dimension
        min_dim = min(highres.shape[0], highres.shape[1], preds.shape[0], preds.shape[1])
        ## Ensure win_size is at most min_dim and at least 3 (since SSIM requires an odd number ≥3)
        win_size = min(min_dim, 7)
        win_size = max(win_size, 3)  ## Ensure it's at least 3
        win_size = win_size - 1 if win_size % 2 == 0 else win_size  ## Ensure odd
        print("Using win_size:", win_size)  ## Debugging step
        print("High-res shape:", highres.shape)
        print("Preds shape:", preds.shape)
        ssim_list.append(ssim(highres,preds,channel_axis=-1,win_size=win_size, data_range=1.0))
        print("ssim",ssim(highres,preds,channel_axis=-1,win_size=win_size, data_range=1.0))
        tag = tag+1
    save_psnr_ssim_data(psnr_list, ssim_list, tag, filebase, base_filename="quality")

def save_psnr_ssim_data_old(psnr_list, ssim_list, tag, filebase, base_filename="results"):
    """
    Saves PSNR and SSIM values to a uniquely named file (name_timestamp.txt).
    Args:
        psnr_list: A list of PSNR values.
        ssim_list: A list of SSIM values.
        tag: The upper bound of the range (exclusive).
        base_filename: The base filename (e.g., "quality").
    """
    filename = f"{filebase+base_filename}_{str(get_timestamp())}.txt"  ## Create unique filename
    try:
        with open(filename, "w") as f:
            for x in range(0, tag - 1):
                f.write(f"{x} {psnr_list[x]} {ssim_list[x]}\n")        ## Write to file
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def save_psnr_ssim_data(psnr_list, ssim_list, tag, filebase, base_filename="results"):
    """
    Saves PSNR and SSIM values to a uniquely named file (name_timestamp.txt) and creates a summary file.

    Args:
        psnr_list: A list of PSNR values.
        ssim_list: A list of SSIM values.
        tag: The upper bound of the range (exclusive).
        filebase: a string to prepend to the filename
        base_filename: The base filename (e.g., "quality").
    """
    filename = f"{filebase+base_filename}_{str(get_timestamp())}.txt"
    summary_filename = f"{filebase+base_filename}_summary_{str(get_timestamp())}.txt" #create a summary filename.

    try:
        with open(filename, "w") as f:
            for x in range(0, tag - 1):
                f.write(f"{x} {psnr_list[x]} {ssim_list[x]}\n")
        print(f"Data saved to {filename}")

        ##Neil Create summary file
        with open(summary_filename, "w") as f_summary:
            f_summary.write("PSNR Summary:\n")
            f_summary.write(f"  Min: {min(psnr_list[:tag-1])}\n")
            f_summary.write(f"  Mean: {np.mean(psnr_list[:tag-1])}\n")
            f_summary.write(f"  Max: {max(psnr_list[:tag-1])}\n")
            f_summary.write("SSIM Summary:\n")
            f_summary.write(f"  Min: {min(ssim_list[:tag-1])}\n")
            f_summary.write(f"  Mean: {np.mean(ssim_list[:tag-1])}\n")
            f_summary.write(f"  Max: {max(ssim_list[:tag-1])}\n")
        print(f"Summary saved to {summary_filename}")

    except Exception as e:
        print(f"Error saving data: {e}")
#################################################### UTILITY FUNCTIONS #########################
def dataset_to_numpy(dataset):
    """ change from loaded dataset to numpy arrays
        Used to change BreastMNIST dataset for SVM analysis
        In this way we only need one data loader for all analysis types
    """
    ## set up the interim structures to allow concatenation across batches
    x_list = []
    y_list = []
    ## loop through dataset
    for batch in dataset:
        ## Unpack the batch into features (x) and labels (y)
        x_batch, y_batch = batch
        ## Append the batches to the lists
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    ## Concatenate all batches into single NumPy arrays, one for x and one for y
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y #x and y as numpy arrays

def get_timestamp():
    """ Gets timestamp. NB specifically compatible with inclusion in filenames
    """
    ## get current datetime
    now = datetime.datetime.now()
    ## reformat it into a timestamp with year, month, day and time in hours, minutes and seconds
    ## seconds added to avoid overwriting for short hyperparameter selection runs
    return now.strftime("%Y_%m_%d_at_%H%M%S") #timestamp

########################################### GRAPHING, SAVING and ANALYSIS #########################
def graph_and_save(history,summary,parameter,filebase,skip=0):
    """ this version calls the two functions together
       summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary
       parameter is hyperparameter store
       skip allows later start point for graphs (default is 0).
       all data goes to files whatever
    """
    graph(history,summary,parameter,skip)
    ## dump history metrics to excel and model and hyper parameters summary
    ## to text file both with same timestamp in names
    run_summary = history_to_excel(history,
                                   str(summary),
                                   parameter,
                                   filebase)
    print("Files saved:",run_summary[0],run_summary[1])
    return run_summary # [filename_h,filename_s,run_result,parameter]

def graph(history, summary, parameter, skip=0):
    """summarize history for accuracy
        history is full history of metrics for all epochs
        summary is model summary (not used, but passed for consistency)
        parameter is hyperparameter store
        skip allows later start point for graphs (default is 0).
    """
    keys    = list(history.history.keys())
    summary = str(summary)
    graph_type = "two"
    if len(keys) == 2:
        graph_type = "one"
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    ## Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs[skip:], history.history[keys[1]][skip:])  ## Loss
    if graph_type == "two":
        plt.plot(epochs[skip:], history.history[keys[3]][skip:])  ## Validation loss
    plt.title('model loss [lr=' + str(parameter.learning_rate) + ']')  ## Correct title
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if graph_type == "two":
        plt.legend(['train', 'val'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    ## Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs[skip:], history.history[keys[0]][skip:])  ## Accuracy
    if graph_type == "two":
        plt.plot(epochs[skip:], history.history[keys[2]][skip:])  ## Validation accuracy
    plt.title('model accuracy [lr=' + str(parameter.learning_rate) + ']')  ## Correct title
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')
    if graph_type == "two":
        plt.legend(['train', 'val'], loc='lower right')
    else:
        plt.legend(['train'], loc='lower right')
    plt.show()

def graph_compare(file1,file2,type_flag='accuracy',index_limit=-1,skip=-1):
    """ allows display of two model runs from metrics files on a single plot
        the flag type controls which metrics to display
        the index_limit and skip can be used to restrict the range displayed
        they default to displaying accuracy for the full range
    """
    if type_flag in ['loss','accuracy']:
        ## read in the data
        data1 = pd.read_excel(file1)
        data2 = pd.read_excel(file2)
        ## Both datasets have the same columns
        columns = [item for item in data1.columns if item != 'epoch']
        if type_flag == 'accuracy':
            columns.remove('loss')
            columns.remove('val_loss')
        else:
            columns.remove('acc')
            columns.remove('val_acc')
        ## slice both dataframes to include only rows up to the chosen range
        if index_limit > 0:
            if len(data1) > index_limit:
                data1 = data1.iloc[:index_limit]
                data2 = data2.iloc[:index_limit]
        if skip > 0:
            if len(data1) > skip:
                data1 = data1.iloc[skip:]
                data2 = data2.iloc[skip:]
        ## define line styles for Model 1 (blue) and Model 2 (green)
        line_styles_model1 = ['solid', 'dashed']  # Model 1 styles
        line_styles_model2 = ['solid', 'dashed']  # Model 2 styles can be different
        ## create the plot
        plt.figure(figsize=(12, 8))
        ## plot comparisons for all chosen variables
        for i,col in enumerate(columns):
            plt.plot(data1[col], label=f'Model 1 '+col, color='blue', \
                    linestyle=line_styles_model1[i % len(line_styles_model1)])
            plt.plot(data2[col], label=f'Model 2 '+col, color='green', \
                    linestyle=line_styles_model2[i % len(line_styles_model2)])
        plt.title(f'Comparison of {type_flag}')
        plt.xlabel('Epoch')
        plt.ylabel(type_flag)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print('Graph metric not recognised')
    ## no return

def history_to_excel(history,summary,parameter,filebase):
    """ puts history metrics into unique excel file
        expanded list of parameters that are handled
        further version could take all of the paramter entries and autoadd to file
    """
    keys = list(history.history.keys())
    ## check to see whether val_ variants are provided
    column_order = []
    column_order.append('epoch')
    for item in keys:
        column_order.append(item)
    ## convert history which is dictionary structure to a DataFrame
    metrics_df = pd.DataFrame(history.history)
    ## add an epoch column to the dataframe for ease of access
    metrics_df['epoch'] = metrics_df.index + 1
    ## then organise the rest of the dataframe ready for saving
    metrics_df = metrics_df[column_order]
    print("metrics_df",keys,column_order)
    ## write dataframe to filename formed by appending timestr to filebase
    timestr    = get_timestamp()
    filename_h = filebase+'metrics_'+timestr+'.xlsx'
    metrics_df.to_excel(filename_h,index=False)
    ## now open the summary text file with matching timestamp
    filename_s = filebase+'summary_'+timestr+'.txt'
    ## write the parameter text to the file
    with open(filename_s, "w") as file:
        file.write(parameter.list_parameters()+summary)
    ## now construct the run result structure with calculated metrics
    run_result = get_run_metrics(metrics_df,
                                 min_loss=('loss', 'min'),
                                 max_acc=('acc', 'max'),
                                 last_loss=('loss', 'last'),
                                 last_acc=('acc', 'last'),
                                 min_val_loss=('val_loss', 'min'),
                                 max_val_acc=('val_acc', 'max'),
                                 last_val_loss=('val_loss', 'last'),
                                 last_val_acc=('val_acc', 'last'),
                                 var_loss=('loss', 'var'),
                                 var_acc=('acc', 'var'))
    ## added parameter to return results
    return [filename_h,filename_s,run_result,parameter] #filenames

def hyper_process(history,_,parameter):
    """ flexibly reads history and writes to dataframe to simplify analysis
        packages a return structure of runresult dataframe and parameter set
        expanded list of parameters that are handled
    """
    keys = list(history.history.keys())
    ## organise the dataframe columns
    column_order = []
    column_order.append('epoch')
    for item in keys:
        column_order.append(item)
    ## convert history which is dictionary structure to a DataFrame
    metrics_df = pd.DataFrame(history.history)
    ## add an epoch column to the dataframe for ease of access
    metrics_df['epoch'] = metrics_df.index + 1
    ## organise the dataframe columns
    metrics_df = metrics_df[column_order]
    ## now construct the run result structure with calculated metric
    if len(column_order) > 2:
        ## val values can be calculated
        run_result = get_run_metrics(metrics_df,
                                    min_loss=('loss', 'min'),
                                    max_acc=('acc', 'max'),
                                    last_loss=('loss', 'last'),
                                    last_acc=('acc', 'last'),
                                    min_val_loss=('val_loss', 'min'),
                                    max_val_acc=('val_acc', 'max'),
                                    last_val_loss=('val_loss', 'last'),
                                    last_val_acc=('val_acc', 'last'),
                                    var_loss=('loss', 'var'),
                                    var_acc=('acc', 'var'))
    else:
        ## set val values to default
        run_result = get_run_metrics(metrics_df,
                                    min_loss=('loss', 'min'),
                                    max_acc=('acc', 'max'),
                                    last_loss=('loss', 'last'),
                                    last_acc=('acc', 'last'),
                                    var_loss=('loss', 'var'),
                                    var_acc=('acc', 'var'))
    ## added parameter to return results
    hyper_history = ["","",run_result,parameter]
    return hyper_history ## ["","",run_result,parameter] to mirror history_to_excel returns

def analyse_run(run_list,selection,filebase):
    """ take run results and analyse
        added filebase param to allow saving of data to file 27122024
    """
    # Extract data into a flat structure
    flat_data = []
    selection = str(selection)
    for entry in run_list:
        ## need to flatten structure for both runresult and parameter
        metrics_file, summary_file, result,parameter = entry
        flat_data.append({
            'metrics_file': metrics_file,
            'summary_file': summary_file,
            # RunResult attributes
            'min_loss': result.min_loss,
            'max_acc': result.max_acc,
            'last_loss': result.last_loss,
            'last_acc': result.last_acc,
            'min_val_loss': result.min_val_loss,
            'max_val_acc': result.max_val_acc,
            'last_val_loss': result.last_val_loss,
            'last_val_acc': result.last_val_acc,
            'var_loss': result.var_loss,
            'var_acc': result.var_acc,
            # HyperParameters attributes
            'learning_rate': parameter.learning_rate,
            'batch_size': parameter.batch_size,
            'num_epochs': parameter.num_epochs,
            'num_filter': parameter.num_filter,
            'strides': parameter.strides,
            'padding':parameter.padding,
            'dropout_rate': parameter.dropout_rate,
            'layers': parameter.layers,
            'optimise': parameter.optimise,
            'loss': parameter.loss,
            'activation': parameter.activation,
        })
    # Convert to DataFrame
    run_df = pd.DataFrame(flat_data)
    ## added save to excel 27122024
    timestr    = get_timestamp() #' may pass this in as param to match other filenames
    filename_r = filebase+'run_'+timestr+'.xlsx'
    run_df.to_excel(filename_r,index=False)
    ## Select the run with the smallest min_loss
    min_loss_run = run_df.loc[run_df['min_loss'].idxmin()]
    ## Select the run with the largest max_acc
    max_acc_run = run_df.loc[run_df['max_acc'].idxmax()]
    ## Select the runs that satisfies the selected criteria
    best_run = run_df.loc[(run_df['min_loss'] == run_df['min_loss'].min()) &
                    (run_df['max_acc'] == run_df['max_acc'].max())]
    if len(best_run) == 0:
        print("No single run matches both objectives, so individually")
        print("Run with the smallest min_loss:")
        print(min_loss_run)
        print("\nRun with the largest max_acc:")
        print(max_acc_run)
    ## Select the runs that satisfy further selected criteria
    ## second best run is the one that maximises validation accuracy and
    ## where maximum accuracy is greater or equal to last accuracy, so plateau or increasing
    best_run2 = run_df.loc[(run_df['max_acc'] == run_df['max_acc'].max()) &
                    (run_df['max_acc'] >= run_df['last_acc'])]
    ## third best run is the one that mimimises validation loss and
    ## where maximum accuracy is less than or equal to last loss, so plateau or falling
    best_run3 = run_df.loc[(run_df['min_loss'] == run_df['min_loss'].min()) &
                    (run_df['min_loss'] <= run_df['last_loss'])]
    return run_df,best_run,best_run2,best_run3

def analyse_hyperparameters(run_df):
    """ analyse hyperparameters using several techniques to gauge their impact
    """
    ## Prepare the input analysis data with hyperparameters as features
    ## doesnt support loss or optimise as they are not numeric values (yet)
    X = run_df[['learning_rate', 'num_epochs', 'num_filter','strides','layers',\
                'dropout_rate','batch_size']]  # Hyperparameters
    y = run_df['max_acc']  ## Metric to predict should this be accuracy or loss?
    ## Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ## Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    ## Evaluate predicted versus actual
    y_pred = model.predict(X_test)
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    ## calculate coefficients for each to understand impact
    coef = pd.DataFrame({'Hyperparameter': X.columns, 'Coefficient': model.coef_})
    ## Fit Random Forest Regressor for max_acc
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, run_df['max_acc'])
    ## calculate feature importance
    feature_importance = pd.DataFrame({
        'Hyperparameter': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return feature_importance,coef

def process_best_run(best_run):
    """ process best_run
    """
    ##print(best_run)
    hp_fields = [f.name for f in fields(HyperParameters)]
    for instance in best_run:
        if instance in hp_fields:
            print(instance,":",best_run[instance].iloc[0])
    ##should be able t do this costruction of parameter more flexibly
    parameter = HyperParameters(learning_rate=best_run['learning_rate'].iloc[0],
                                batch_size=best_run['batch_size'].iloc[0],
                                num_epochs=best_run['num_epochs'].iloc[0],
                                num_filter=best_run['num_filter'].iloc[0],
                                layers=best_run['layers'].iloc[0],
                                dropout_rate=best_run['dropout_rate'].iloc[0],
                                strides=best_run['strides'].iloc[0],
                                padding=best_run['padding'].iloc[0],
                                optimise=best_run['optimise'].iloc[0],
                                loss=best_run['loss'].iloc[0])
    parameter.save_excel("param_"+str(get_timestamp())+".xlsx")

################################# AMLS2 functions ################################################

class ResizeLayer(Layer):
    """ resize layer"""
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs, **kwargs): ## added kwargs
        return tf.image.resize(inputs, self.target_size)

def display_lr_hr_pairs(dataset, num_samples=5):
    """
    Displays LR-HR image pairs from a dataset without using OpenCV.

    Parameters:
        dataset (tf.data.Dataset): The dataset containing (LR, HR) pairs.
        num_samples (int): Number of pairs to display.
    """
    ## Get a batch of images
    lowres_batch, highres_batch = next(iter(dataset))
    ## Convert tensors to NumPy for visualization
    lowres_batch = lowres_batch.numpy()
    highres_batch = highres_batch.numpy()
    # Plot the images
    plt.figure(figsize=(10, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(lowres_batch[i])  ## Show LR image
        plt.axis("off")
        plt.title("Low-Res")
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(highres_batch[i])  ## Show HR image
        plt.axis("off")
        plt.title("High-Res")
    plt.show()

#################################################### DATA LOADING ##############################

def load_image_pair(lr_path, hr_path, upscale_factor):
    """
    Loads a low-resolution (LR) and high-resolution (HR) image pair as tensors.
    Now add in explicit upscale factor
    """
    ## Load LR image
    lr = tf.io.read_file(lr_path)
    lr = tf.image.decode_png(lr, channels=3)
    lr = tf.image.convert_image_dtype(lr, tf.float32)  # Normalize to [0,1]
    ## Resize LR image to MobileNet input size
    lr = tf.image.resize(lr, [IMG_SIZE, IMG_SIZE])
    ## Load HR image
    hr = tf.io.read_file(hr_path)
    hr = tf.image.decode_png(hr, channels=3)
    hr = tf.image.convert_image_dtype(hr, tf.float32)  # Normalize to [0,1]
    ## Do we resize HR image to keep its original resolution for training
    ## Resize HR to 4x LR size
    hr = tf.image.resize(hr, [IMG_SIZE * upscale_factor, IMG_SIZE * upscale_factor])
    return lr, hr  ## Return both as tensors

#######################################################################################
def plot_results(lowres, preds):
    """
    Displays low-resolution image and super-resolution image
    """
    plt.figure(figsize=(12, 6))
    ## Ensure pixel values are within valid range [0,1]
    lowres = np.clip(lowres, 0, 1)
    preds  = np.clip(preds, 0, 1)
    plt.subplot(1, 2, 1)
    plt.imshow(lowres)
    plt.title("Low-resolution")
    plt.subplot(1, 2, 2)
    plt.imshow(preds)
    plt.title("Prediction (Super-Resolution)")
    plt.show()

## Define perceptual loss outside the training loop
# Load pre-trained VGG19 model (only once, outside the function)
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# input_shape should match your image dimensions
# Extract features from a specific layer
loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
loss_model.trainable = False  # Freeze VGG19 weights

def swish(x):
    """ swish function
    """
    return x * tf.keras.activations.sigmoid(x)

def perceptual_loss(y_true, y_pred):
    """
    Calculates the perceptual loss using VGG19 features.
    Args:
        y_true: Ground truth high-resolution image.
        y_pred: Predicted high-resolution image.
    Returns:
        Perceptual loss value.
    """
    ## Resize y_true and y_pred to (224, 224) before passing to loss_model
    y_true = tf.image.resize(y_true, (224, 224))
    y_pred = tf.image.resize(y_pred, (224, 224))
    ## Calculate perceptual loss using the pre-loaded loss_model
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))

def srresnet_base(num_res_blocks: int = 16,dropout_rate=0.0):
    """ Creates SRResNet model - now with added SE blocks
    Parameters
    ----------
    num_res_blocks: int - Number of residual blocks in the model - Default=16
    Returns
    -------
    SRResNet Model object.
    """
    def PReLU_activation(name):
        """ PReLU """
        return PReLU(Constant(value=0.25), shared_axes=[1,2], name=name)

    def se_block(input_tensor, ratio=16):
        """Squeeze-and-Excitation block for feature enhancement."""
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // ratio, activation="relu")(se)
        se = Dense(filters, activation="sigmoid")(se)
        return Multiply()([input_tensor, se])

    def residual_srresnet_block(layer_input, filters, block_number,dropout_rate=0.0):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same',\
                   name=f"conv_res_{block_number}_1")(layer_input)
        d = PReLU_activation(f"prelu_res_{block_number}")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_1")(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same',\
                   name=f"conv_res_{block_number}_2")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_2")(d)
        if dropout_rate > 0.0:
            d = Dropout(dropout_rate)(d)
        # Add SE block here
        d = se_block(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, scale, i):
        """ upsample block
        """
        u = Conv2D(256, kernel_size=3, strides=1, padding='same',\
                   name=f"conv_up_{i}")(layer_input)
        # Wrap depth_to_space in a Lambda layer and added scale parameter rather than 2
        u = Lambda(lambda x: depth_to_space(x, scale), name=f"pix_shuf_{i}")(u)
        return PReLU_activation(name=f"prelu_up_{i}")(u)

    ## Model Construction
    lr_image = Input(shape=(None, None, 3))
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same',\
                name="Conv_ip")(lr_image)
    c1 = PReLU_activation(name="prelu_ip")(c1)
    r  = residual_srresnet_block(c1, 64, 0)
    for i in range(1,num_res_blocks):
        r = residual_srresnet_block(r, 64, i)
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same',\
                name="conv_out")(r)
    c2 = BatchNormalization(momentum=0.8, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])
    u1 = upsample_block(c2, 2, 1)
    u2 = upsample_block(u1, 2, 2)
    c3 = Conv2D(3, kernel_size=9, strides=1, padding='same',\
                activation="sigmoid", name="conv_final")(u2)
    return Model(lr_image, c3, name="SRResNet")

def residual_in_residual_dense_block(layer_input, filters, block_number, kernel_size, padding):
    d = layer_input
    for i in range(3):
        d = Conv2D(filters, kernel_size=kernel_size, padding=padding, activation="relu", name=f"rrdb_{block_number}_{i}")(d)
        d = Add()([d, layer_input])
    return d

def rrdb_block(layer_input, filters, block_number, kernel_size, padding):
    """Conceptual RRDB block."""
    d = layer_input
    dense_layers = []  ## Store outputs of dense layers
    for i in range(3): ## Multiple dense layers
        d_temp = Conv2D(filters, kernel_size=kernel_size, padding=padding, activation="relu", name=f"RRDB_conv_{block_number}_{i}")(d)
        dense_layers.append(d_temp)
        d = Concatenate()([d, d_temp])  ## Concatenate features
    d = Conv2D(filters, kernel_size=1, padding='same', name=f"RRDB_conv_out_{block_number}")(d)  # Output convolution
    d = Add()([layer_input, d])  ## Short residual connection
    ## Residual-in-Residual
    d_outer = d
    for i in range(2): # multiple residual blocks inside the RRDB
        d_temp = Conv2D(filters, kernel_size=kernel_size, padding=padding, activation="relu", name=f"RRDB_inner_conv_{block_number}_{i}")(d_outer)
        d_outer = Add()([d_outer,d_temp])
    return d_outer

def srresnet_tune(item):
    """ Creates SRResNet model with SE blocks and enhanced tunability
    Parameters
    ----------
        item: Hyperparameters        
    Returns
    -------
        SRResNet Model object.
    """
    from tensorflow.keras import layers
    ## initialise parameters
    ## model parameters
    activation               = str(item.activation)
    loss                     = str(item.loss)
    optimizer_choice         = str(item.optimise)
    learning_rate            = float(item.learning_rate)
    ## CNN defaults
    padding                  = str(item.padding)
    strides                  = int(item.strides)
    ## width of CNN
    num_filter               = int(item.num_filter)               
    ## Number of residual blocks in the model - depth of CNN
    num_layers               = int(item.layers)                   
    kernel_size              = int(item.kernel_size)
    ## Upscaling factor e.g. 2,4
    scale                    = int(item.scale) 
    ## Parameters for batch normalization                   
    momentum                 = float(item.momentum)
    epsilon                  = float(item.epsilon)
    lr_image                 = Input(shape=(None, None, 3))
    ## parameter not currently used but available
    ##   dropout_rate = float(item.dropout_rate)

    def activation_layer(activation,name):
        """ activation layer with activation and name parameters"""
        if activation == 'relu':
            return layers.ReLU(name=name)
        if activation == 'leakyrelu':
            return layers.LeakyReLU(alpha=0.2, name=name)
        if activation == 'prelu':
            return PReLU(Constant(value=0.25), shared_axes=[1, 2], name=name)

    def se_block(input_tensor, ratio=16):
        """Squeeze-and-Excitation block for feature enhancement """
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // ratio, activation="relu")(se)
        se = Dense(filters, activation="sigmoid")(se)
        return Multiply()([input_tensor, se])

    def residual_srresnet_block(layer_input, filters, block_number):
        """Residual block described in Lim et al [ ]"""
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_res_{block_number}_1")(layer_input)
        d = BatchNormalization(momentum=momentum, epsilon=epsilon,\
                               name=f"BN_res_{block_number}_1")(d)
        d = activation_layer(activation,f"activation_res_{block_number}_1")(d)
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_res_{block_number}_2")(d)
        d = BatchNormalization(momentum=momentum, epsilon=epsilon,\
                               name=f"BN_res_{block_number}_2")(d)
        ## putting se block here allows feature refinement
        d = se_block(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def residual_srresnet_block_new(layer_input, filters, block_number):
        d = residual_in_residual_dense_block(layer_input, filters, block_number, kernel_size, padding)
        d = se_block(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(activation,layer_input, scale, i):
        """ upsample block to increase the spatial resolution (height and width) 
            of the input feature maps """
        u = Conv2D(256, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_up_{i}")(layer_input)
        if scale == 2:  ## can use depth to space efficent process with pixel shuffle
            u = Lambda(lambda x: depth_to_space(x, scale), name=f"pix_shuf_{i}")(u)
        else:
            ## scale other than 2 needs bilinear interpolation
            u = UpSampling2D(size=(scale, scale), interpolation='bilinear',\
                             name = f"up_sample_{i}")(u)
        return activation_layer(activation,name=f"activation_up_{i}")(u)

    ## Model Construction
    ## kernel size = 9. 128 was 64
    c1 = Conv2D(128, kernel_size=9, strides=strides, padding=padding, name="Conv_ip")(lr_image)
    print("activation",activation)
    c1 = activation_layer(activation,name="activation_ip")(c1)
    r  = residual_srresnet_block(c1, num_filter, 0)
    ## use num_layers from item
    for i in range(1, num_layers):
        r = residual_srresnet_block(r, num_filter, i)
    ## 128 was 64
    c2 = Conv2D(128, kernel_size=kernel_size, strides=strides, padding=padding,\
                name="conv_out")(r)
    c2 = BatchNormalization(momentum=momentum, epsilon=epsilon, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])
    u1 = upsample_block(activation,c2, scale, 1)
    if scale == 4:
        u2 = upsample_block(activation,u1, 2, 2)
        u2 = upsample_block(activation,u2, 2, 3)
    ## 3 to match channels. kernel = 9 
    c3 = Conv2D(3, kernel_size=9, strides=strides, padding=padding, activation="sigmoid",\
                name="conv_final")(u1 if scale == 2 else u2)
    model = Model(lr_image, c3, name="SRResNet")
    ## optimizer setting and model compile
    if optimizer_choice == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer_choice == 'AdamW':
        optimizer = AdamW(learning_rate=learning_rate)
    if loss == 'ssim_loss':
        model.compile(loss=ssim_loss_plus,\
                      optimizer=optimizer,\
                      metrics=['acc'])
    if loss == 'psnr_loss':
        model.compile(loss=psnr_loss,\
                      optimizer=optimizer,\
                      metrics=['acc'])
    return model

def srresnet_plus(item):
    """Creates SRResNet model with RRDB blocks and enhanced tunability."""
    ## initialise parameters
    ## model parameters
    activation               = str(item.activation)
    loss                     = str(item.loss)
    optimizer_choice         = str(item.optimise)
    learning_rate            = float(item.learning_rate)
    ## CNN defaults
    padding                  = str(item.padding)
    strides                  = int(item.strides)
    ## width of CNN
    num_filter               = int(item.num_filter)               
    ## Number of residual blocks in the model - depth of CNN
    num_layers               = int(item.layers)                   
    kernel_size              = int(item.kernel_size)
    ## Upscaling factor e.g. 2,4
    scale                    = int(item.scale) 
    ## Parameters for batch normalization                   
    momentum                 = float(item.momentum)
    epsilon                  = float(item.epsilon)
    lr_image                 = Input(shape=(None, None, 3))
    ## parameter not currently used but available
    ##   dropout_rate = float(item.dropout_rate)

    def activation_layer(activation, name):
        if activation == 'relu':
            return layers.ReLU(name=name)
        if activation == 'leakyrelu':
            return layers.LeakyReLU(alpha=0.2, name=name)
        if activation == 'prelu':
            return PReLU(Constant(value=0.25), shared_axes=[1, 2], name=name)

    def se_block(input_tensor, ratio=16):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // ratio, activation="relu")(se)
        se = Dense(filters, activation="sigmoid")(se)
        return Multiply()([input_tensor, se])

    def residual_srresnet_block(layer_input, filters, block_number):
        d = rrdb_block(layer_input, filters, block_number, kernel_size, padding)
        d = se_block(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(activation, layer_input, scale, i):
        u = Conv2D(256, kernel_size=kernel_size, strides=strides, padding=padding, name=f"conv_up_{i}")(layer_input)
        if scale == 2:
            u = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name=f"pix_shuf_{i}")(u)
        else:
            u = UpSampling2D(size=(scale, scale), interpolation='bilinear', name=f"up_sample_{i}")(u)
        return activation_layer(activation, name=f"activation_up_{i}")(u)

    c1 = Conv2D(128, kernel_size=9, strides=strides, padding=padding, name="Conv_ip")(lr_image)
    c1 = activation_layer(activation, name="activation_ip")(c1)
    r  = residual_srresnet_block(c1, num_filter, 0)

    for i in range(1, num_layers):
        r = residual_srresnet_block(r, num_filter, i)

    c2 = Conv2D(128, kernel_size=kernel_size, strides=strides, padding=padding, name="conv_out")(r)
    c2 = BatchNormalization(momentum=momentum, epsilon=epsilon, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])
    u1 = upsample_block(activation, c2, scale, 1)

    if scale == 4:
        u2 = upsample_block(activation, u1, 2, 2)
        u2 = upsample_block(activation, u2, 2, 3)

    c3 = Conv2D(3, kernel_size=9, strides=strides, padding=padding, activation="sigmoid", name="conv_final")(u1 if scale == 2 else u2)
    model = Model(lr_image, c3, name="SRResNet")

    if optimizer_choice == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer_choice == 'AdamW':
        optimizer = AdamW(learning_rate=learning_rate)
    if loss == 'ssim_loss':
        model.compile(loss=ssim_loss_plus, optimizer=optimizer, metrics=['acc'])
    if loss == 'psnr_loss':
        model.compile(loss=psnr_loss, optimizer=optimizer, metrics=['acc'])
    return model

def srresnet_tune_2(item):
    """ Creates SRResNet model with SE blocks and enhanced tunability
    Parameters
    ----------
        item: Hyperparameters        
    Returns
    -------
        SRResNet Model object.
    """
    from tensorflow.keras import layers
    ## initialise parameters
    ## model parameters
    activation               = str(item.activation)
    loss                     = str(item.loss)
    optimizer_choice         = str(item.optimise)
    learning_rate            = float(item.learning_rate)
    ## CNN defaults
    padding                  = str(item.padding)
    strides                  = int(item.strides)
    ## width of CNN
    num_filter               = int(item.num_filter)               
    ## Number of residual blocks in the model - depth of CNN
    num_layers               = int(item.layers)                   
    kernel_size              = int(item.kernel_size)
    ## Upscaling factor e.g. 2,4
    scale                    = int(item.scale) 
    ## Parameters for batch normalization                   
    momentum                 = float(item.momentum)
    epsilon                  = float(item.epsilon)
    lr_image                 = Input(shape=(None, None, 3))
    ## parameter not currently used but available
    ##   dropout_rate = float(item.dropout_rate)

    def activation_layer(activation,name):
        """ activation layer with activation and name parameters"""
        if activation == 'relu':
            return layers.ReLU(name=name)
        if activation == 'leakyrelu':
            return layers.LeakyReLU(alpha=0.2, name=name)
        if activation == 'prelu':
            return PReLU(Constant(value=0.25), shared_axes=[1, 2], name=name)

    def se_block(input_tensor, ratio=16):
        """Squeeze-and-Excitation block for feature enhancement """
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // ratio, activation="relu")(se)
        se = Dense(filters, activation="sigmoid")(se)
        return Multiply()([input_tensor, se])

    def residual_srresnet_block(layer_input, filters, block_number):
        """Residual block described in Lim et al [ ]"""
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_res_{block_number}_1")(layer_input)
        d = BatchNormalization(momentum=momentum, epsilon=epsilon,\
                               name=f"BN_res_{block_number}_1")(d)
        d = activation_layer(activation,f"activation_res_{block_number}_1")(d)
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_res_{block_number}_2")(d)
        d = BatchNormalization(momentum=momentum, epsilon=epsilon,\
                               name=f"BN_res_{block_number}_2")(d)
        ## putting se block here allows feature refinement
        d = se_block(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(activation,layer_input, scale, i):
        """ upsample block to increase the spatial resolution (height and width) 
            of the input feature maps """
        u = Conv2D(256, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_up_{i}")(layer_input)
        if scale == 2:  ## can use depth to space efficent process with pixel shuffle
            u = Lambda(lambda x: depth_to_space(x, scale), name=f"pix_shuf_{i}")(u)
        else:
            ## scale other than 2 needs bilinear interpolation
            u = UpSampling2D(size=(scale, scale), interpolation='bilinear',\
                             name = f"up_sample_{i}")(u)
        return activation_layer(activation,name=f"activation_up_{i}")(u)

    ## Model Construction
    ## kernel size = 9. 128 was 64
    c1 = Conv2D(128, kernel_size=9, strides=strides, padding=padding, name="Conv_ip")(lr_image)
    print("activation",activation)
    c1 = activation_layer(activation,name="activation_ip")(c1)
    r  = residual_srresnet_block(c1, num_filter, 0)
    ## use num_layers from item
    for i in range(1, num_layers):
        r = residual_srresnet_block(r, num_filter, i)
    ## 128 was 64
    c2 = Conv2D(128, kernel_size=kernel_size, strides=strides, padding=padding,
                name="conv_out")(r)
    c2 = BatchNormalization(momentum=momentum, epsilon=epsilon, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])

    # Upsampling (adjust based on your desired scale factor and output size)
    # u1 = upsample_block(activation, c2, scale, 1)  # Original upsampling call
    # if scale == 4:  
    #     u2 = upsample_block(activation, u1, 2, 2)
    #     u2 = upsample_block(activation, u2, 2, 3)
        
    # Instead of the above, or after the above, add resizing:
    upscaled_output = layers.Resizing(448, 448)(c2)  # Resize to 448x448

    ## 3 to match channels. kernel = 9 
    c3 = Conv2D(3, kernel_size=9, strides=strides, padding=padding, activation="sigmoid",
                name="conv_final")(upscaled_output)  # Use resized output
    model = Model(lr_image, c3, name="SRResNet")
    ## optimizer setting and model compile
    if optimizer_choice == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer_choice == 'AdamW':
        optimizer = AdamW(learning_rate=learning_rate)
    if loss == 'ssim_loss':
        model.compile(loss=ssim_loss_plus,\
                      optimizer=optimizer,\
                      metrics=['acc'])
    if loss == 'psnr_loss':
        model.compile(loss=psnr_loss,\
                      optimizer=optimizer,\
                      metrics=['acc'])
    return model

def edsr(num_filters: int = 64, num_res_blocks: int = 16):
    """ Creates an EDSR model.
    Parameters
    ----------
    num_filters: int          Number of filters per convolution layer     Default=64
    num_res_blocks: int       Number of residual blocks in the model      Default=16
    Returns
    -------
        EDSR Model object
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize      = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize    = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle  = lambda x: depth_to_space(x, 2)

    def residual_edsr_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same',\
                   activation='relu', name=f"conv_res_{block_number}_1")(layer_input)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same',\
                   name=f"conv_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, i):
        u = Conv2D(num_filters*4, kernel_size=3, strides=1, padding='same',\
                   name=f"conv_up_{i}")(layer_input)
        u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u)
        return u

    ## Model Construction
    x_in  = Input(shape=(None, None, 3), name="LR Batch")
    x     = Lambda(normalize, name="normalize_input")(x_in)
    x = r = Conv2D(num_filters, 3, padding='same', name="Conv_ip")(x)
    for i in range(num_res_blocks):
        r = residual_edsr_block(r, num_filters, i)
    c2    = Conv2D(num_filters, 3, padding='same', name="conv_out")(r)
    c2    = Add(name="add_out")([x, c2])
    u1    = upsample_block(c2, 1)
    u2    = upsample_block(u1, 2)
    c3    = Conv2D(3, 3, padding='same', name="conv_final")(u2)
    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(x_in, x_out, name="EDSR")

def edsr_plus(item):
    """ Creates an EDSR model.
    Parameters
    ----------
        item: Hyperparameters        
    Returns
    -------
        EDSR Model object
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize      = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize    = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle  = lambda x: depth_to_space(x, 2)
    ## initialise parameters
    ## model parameters
    activation               = str(item.activation)
    loss                     = str(item.loss)
    optimizer_choice         = str(item.optimise)
    learning_rate            = float(item.learning_rate)
    ## CNN defaults
    padding                  = str(item.padding)
    strides                  = int(item.strides)
    ## width of CNN
    num_filters              = int(item.num_filter)               
    ## Number of residual blocks in the model - depth of CNN
    num_layers               = int(item.layers)                   
    kernel_size              = int(item.kernel_size)
    ## Upscaling factor e.g. 2,4
    scale                    = int(item.scale) 
    ## Parameters for batch normalization                   
    momentum                 = float(item.momentum)
    epsilon                  = float(item.epsilon)
    lr_image                 = Input(shape=(None, None, 3))
    ## parameter not currently used but available
    ##   dropout_rate = float(item.dropout_rate)

    def residual_edsr_block(layer_input, filters, kernel_size,
                            strides, padding, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   activation='relu', name=f"conv_res_{block_number}_1")(layer_input)
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,\
                   name=f"conv_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, num_filters, kernel_size,
                       strides, padding, i):
        u = Conv2D(num_filters*4, kernel_size=kernel_size, strides=strides,\
                   padding=padding,name=f"conv_up_{i}")(layer_input)
        u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u)
        return u

    ## Model Construction
    x_in  = Input(shape=(None, None, 3), name="LR Batch")
    x     = Lambda(normalize, name="normalize_input")(x_in)
    x = r = Conv2D(num_filters, 3, padding=padding, name="Conv_ip")(x)
    ## number of res blocks
    for i in range(num_layers):
        r = residual_edsr_block(r, num_filters,kernel_size,
                                strides,padding,i)
    c2    = Conv2D(num_filters, 3, padding=padding, name="conv_out")(r)
    c2    = Add(name="add_out")([x, c2])
    u1    = upsample_block(c2,num_filters, kernel_size,
                           strides, padding, 1)
    u2    = upsample_block(u1,num_filters, kernel_size,
                           strides, padding, 2)
    c3    = Conv2D(3, 3, padding=padding, name="conv_final")(u2)
    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(x_in, x_out, name="EDSR")

def calculate_psnr(firstImage, secondImage, max_pixel=255.0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        firstImage: The first image (NumPy array).
        secondImage: The second image (NumPy array).
        max_pixel: The maximum possible pixel value. Defaults to 255.0.
    Returns:
        The PSNR value.
    """
    # Ensure images are the same shape
    if firstImage.shape != secondImage.shape:
        raise ValueError("Images must have the same dimensions.")
    ## Convert images to float64 for accurate calculations
    firstImage = firstImage.astype(np.float64)
    secondImage = secondImage.astype(np.float64)
    ## Compute the mean squared error (MSE)
    mse = np.mean((firstImage - secondImage) ** 2)
    ## Handle the case where MSE is zero (perfect match)
    if mse == 0:
        return float('inf')  ## PSNR is infinite for perfect match
    ## Calculate PSNR
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

def ssim_loss(y_true, y_pred):
    """ ssim loss function
    """
    ##print("Shape of y_true:", tf.shape(y_true))  # Print shape of ground truth
    ##print("Shape of y_pred:", tf.shape(y_pred))  # Print shape of prediction
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

def ssim_loss_plus(y_true, y_pred):
    """
    SSIM loss function
    """
    # Convert to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Resize images if they have different sizes
    if y_true.shape[1] != y_pred.shape[1] or y_true.shape[2] != y_pred.shape[2]:
        # Use tf.keras.layers.Resizing instead of tf.image.resize
        y_pred = tf.keras.layers.Resizing(
            height=tf.shape(y_true)[1], width=tf.shape(y_true)[2]
        )(y_pred)
    # Calculate SSIM
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
    # Return SSIM loss
    return 1 - tf.reduce_mean(ssim_value)

def mse_loss(y_true, y_pred):
    """ MSE loss function
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def psnr_loss(y_true, y_pred):
    """ PSNR loss function with normalized return value to improve stability
    """
    mse = mse_loss(y_true, y_pred)
    max_val = 1.0 ## assumes images are normalized.
    psnr = 10.0 * tf.math.log(tf.square(max_val) / mse) / tf.math.log(10.0)
    ## Normalize PSNR to [-1, 1]
    typical_max_psnr = 50.0  ## Adjust this value if needed based on runs
    normalized_psnr = psnr / typical_max_psnr
    return -normalized_psnr ## make it a loss by inverting it.

########################## code holding ##########################
