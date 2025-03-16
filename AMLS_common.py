## Common code to support all tasks of AMLS Assigmnent
## including the loading the MedMNIST data files into tensorflow format
## loading of hyperparameters into single data class structure
## and handling of NN history results: graphing and storing results in files
## Used across both assignment tasks
## import common libraries
## Revision History
## 02122024 Tidy functions & comments (including pylint run)
## 02122024 Split graph and save functions to allow graphing without saving for heavy testing
## 02122024 Update graph function to allow plot start at skip to ease analysis
## 07122024 Updated for more stable use of medmnist library and associated functions
## 09122024 Enhanced dataclass with defaults and list function for saving to file
## 09122024 Add tqdm custom callback
## 15122024 Extended HyperParameter
## 16122024 Again extended HyperParameter e.g. layers, dropout, filter2
## 16122024 Integrated extended analysis code from Hyper script into library to faciitate sharing
## 20122024 Extended parameter again
## 27122024 Comments and modifications for Task B1 CNN Tune
## 31122024 Extended dataclasses and enhanced hyper analysis in combination with model scripts
## 11012025 Added compare graph function and overfitting callback rather than previous manual option
## 16032025 Added AMLS2 base functions, plus minor updates to existing functions

#################################################### LIBRARY IMPORTS ##############################
## standard python libraries
import datetime
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
## import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Add, Layer, Lambda, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, Flatten, UpSampling2D, Dropout, BatchNormalization, PReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge, MeanAbsoluteError
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.nn import depth_to_space

## MedMNIST specific libraries loading all relevant items (updated 07122024)
##import medmnist
##from medmnist import INFO ##, info
## sklearn to allow analysis of hyperparameter choices
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skimage.metrics import structural_similarity as ssim

#################################################### SET UP DATACLASSES ##############################
@dataclass
class HyperParameters:
    """ data class to allow storage and passing set of hyperparameters as structure
    """
    learning_rate: float
    kernel_size: int
    num_epochs: int
    optimise: str
    loss: str
    num_filter: int
    strides: int = 1
    padding: str = "valid"
    dropout_rate: float = 0.2
    layers: int = 3
    default_activation: str = "relu"

    def list_parameters(self):
        """ lists all attributes and values in HyperParameters class
        """
        result = ""   ## initalise result
        # Loop through attributes and get their values
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
        # Ensure the file contains at least one row
        if df_load.empty:
            raise ValueError(f"The file {file_path} is empty.")
        # Convert the first row of the DataFrame into a dictionary
        data_dict = df_load.iloc[0].to_dict()
        # Pass the dictionary as keyword arguments to the dataclass constructor
        return cls(**data_dict)
    
    def save_excel(self, file_path: str):
        """
        Saves the dataclass instance to an Excel file.
        """
        # Convert the dataclass to a dictionary
        data = self.__dict__
        # Convert the dictionary into a pandas DataFrame
        df_save = pd.DataFrame([data])  # Wrap in a list to create a single-row DataFrame
        # Save the DataFrame to Excel
        df_save.to_excel(file_path, index=False)

@dataclass
class RunResult:
    """ data class to allow storage and passing of summary run results as structure
    """
    min_loss: float
    max_acc: float
    last_loss: float
    last_acc: float
    min_val_loss: float
    max_val_acc: float
    last_val_loss: float
    last_val_acc: float
    var_loss: float
    var_acc: float

    def list_runresult(self):
        """ lists all attributes and values in RunResult class
        """
        result = ""   ## initalise result
        # Loop through attributes and get their values
        for field in fields(RunResult):
            attribute_name = field.name
            value = getattr(self, attribute_name)
            result= result+(f"{attribute_name}: {value}"+"\n")
        return result

class TqdmEpochProgress(tf.keras.callbacks.Callback):
    """ simple progress bar
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

#################################################### UTILITY FUNCTIONS ##############################
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


########################################### GRAPHING, SAVING and ANALYSIS ##############################
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

def graph(history,summary,parameter,skip=0):
    """summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary (not used, but passed for consistency)
       parameter is hyperparameter store
       skip allows later start point for graphs (default is 0).
    """
    ## first load the keys supplied as part of history.
    ## used to dynamically set various graph elements
    keys = list(history.history.keys())
    ## work out if there is a single or double graph lines
    graph_type = "two"
    if len(keys) == 2:
        graph_type = "obe"
    ## set the epochs range for use in plot
    epochs = range(1,len(history.history['loss'])+1)
    ## initalise the plot size
    plt.figure(figsize=(12, 5))
    ## set up the first subplot
    plt.subplot(1, 2, 1)
    ## set the first line based on the specific accuracy key supplied
    plt.plot(epochs[skip:],history.history[keys[1]][skip:])
    if graph_type == "two":
        plt.plot(epochs[skip:],history.history[keys[3]][skip:])
    plt.title('model accuracy [lr='+str(parameter.learning_rate)+']')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    ## set the legend depending on whether the graph has one or two lines
    if graph_type == "two":
        plt.legend(['train','val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    ## now set up the second subplot alongside the first
    plt.subplot(1, 2, 2)
    ## summarize history for loss based on supplid loss key
    plt.plot(epochs[skip:],history.history[keys[0]][skip:])
    ## set the legend depending on whether the graph has one or two lines
    if graph_type == "two":
        plt.plot(epochs[skip:],history.history[keys[2]][skip:])
    plt.title('model loss [lr='+str(parameter.learning_rate)+']')
    plt.ylabel(keys[0])
    plt.xlabel('epoch')
    if graph_type == "two":
        plt.legend(['train','val'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    plt.show()
    print("for model\n",str(summary))
    ## no return

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
    run_result = RunResult(min_loss      = metrics_df[column_order[1]].min(),
                           max_acc       = metrics_df[column_order[2]].max(),
                           last_loss     = metrics_df[column_order[1]].iloc[-1],
                           last_acc      = metrics_df[column_order[2]].iloc[-1],
                           min_val_loss  = metrics_df[column_order[3]].min(),
                           max_val_acc   = metrics_df[column_order[4]].max(),
                           last_val_loss = metrics_df[column_order[3]].iloc[-1],
                           last_val_acc  = metrics_df[column_order[4]].iloc[-1],
                           var_loss      = metrics_df[column_order[1]].var(),
                           var_acc       = metrics_df[column_order[2]].var())
    ## added parameter to return results
    return [filename_h,filename_s,run_result,parameter] #filenames

def hyper_process(history,_,parameter):
    """ flexibly reads history and writes to dataframe to simplify analysis
        packages a return structure of runresult dataframe and paramter set
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
        run_result = RunResult(min_loss      = metrics_df[column_order[1]].min(),
                               max_acc       = metrics_df[column_order[2]].max(),
                               last_loss     = metrics_df[column_order[1]].iloc[-1],
                               last_acc      = metrics_df[column_order[2]].iloc[-1],
                               min_val_loss  = metrics_df[column_order[3]].min(),
                               max_val_acc   = metrics_df[column_order[4]].max(),
                               last_val_loss = metrics_df[column_order[3]].iloc[-1],
                               last_val_acc  = metrics_df[column_order[4]].iloc[-1],
                               var_loss      = metrics_df[column_order[1]].var(),
                               var_acc       = metrics_df[column_order[2]].var())
    else:
        ## set val values to default
        run_result = RunResult(min_loss      = metrics_df[column_order[1]].min(),
                               max_acc       = metrics_df[column_order[2]].max(),
                               last_loss     = metrics_df[column_order[1]].iloc[-1],
                               last_acc      = metrics_df[column_order[2]].iloc[-1],
                               min_val_loss  = 99999,
                               max_val_acc   = 0,
                               last_val_loss = 99999,
                               last_val_acc  = 0,
                               var_loss      = metrics_df[column_order[1]].var(),
                               var_acc       = metrics_df[column_order[2]].var())
    ## added parameter to return results
    hyper_history = ["","",run_result,parameter]
    return hyper_history ## ["","",run_result,parameter] to mirror history_to_excel returns

def analyse_run(run_list,selection,filebase):
    """ take run results and analyse
        added filebase param to allow saving of data to file 27122024
    """
    # Extract data into a flat structure
    flat_data = []
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
            'kernel_size': parameter.kernel_size,
            'num_epochs': parameter.num_epochs,
            'num_filter': parameter.num_filter,
            'strides': parameter.strides,
            'padding':parameter.padding,
            'dropout_rate': parameter.dropout_rate,
            'layers': parameter.layers,
            'optimise': parameter.optimise,
            'loss': parameter.loss,
            'default_activation': parameter.default_activation,
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
                'dropout_rate','kernel_size']]  # Hyperparameters
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
                                kernel_size=best_run['kernel_size'].iloc[0], 
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
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

def display_lr_hr_pairs(dataset, num_samples=5):
    """
    Displays LR-HR image pairs from a dataset without using OpenCV.

    Parameters:
        dataset (tf.data.Dataset): The dataset containing (LR, HR) pairs.
        num_samples (int): Number of pairs to display.
    """
    # Get a batch of images
    lowres_batch, highres_batch = next(iter(dataset))

    # Convert tensors to NumPy for visualization
    lowres_batch = lowres_batch.numpy()
    highres_batch = highres_batch.numpy()

    # Plot the images
    plt.figure(figsize=(10, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(lowres_batch[i])  # Show LR image
        plt.axis("off")
        plt.title("Low-Res")

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(highres_batch[i])  # Show HR image
        plt.axis("off")
        plt.title("High-Res")

    plt.show()

#################################################### DATA LOADING ##############################

def load_image_pair(lr_path, hr_path):
    """
    Loads a low-resolution (LR) and high-resolution (HR) image pair as tensors.
    """
    # Load LR image
    lr = tf.io.read_file(lr_path)
    lr = tf.image.decode_png(lr, channels=3)
    lr = tf.image.convert_image_dtype(lr, tf.float32)  # Normalize to [0,1]

    # Resize LR image to MobileNet input size
    lr = tf.image.resize(lr, [IMG_SIZE, IMG_SIZE])

    # Load HR image
    hr = tf.io.read_file(hr_path)
    hr = tf.image.decode_png(hr, channels=3)
    hr = tf.image.convert_image_dtype(hr, tf.float32)  # Normalize to [0,1]
    # Do we resize HR image to keep its original resolution for training
    ##hr = tf.image.resize(hr, [IMG_SIZE, IMG_SIZE])
    hr = tf.image.resize(hr, [IMG_SIZE * UPSCALE_FACTOR, IMG_SIZE * UPSCALE_FACTOR])  # Resize HR to 4x LR size

    return lr, hr  # Return both as tensors

#######################################################################################
def plot_results(lowres, preds):
    """
    Displays low-resolution image and super-resolution image
    """
    plt.figure(figsize=(12, 6))

    # Ensure pixel values are within valid range [0,1]
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
    # Resize y_true and y_pred to (224, 224) before passing to loss_model
    y_true = tf.image.resize(y_true, (224, 224))
    y_pred = tf.image.resize(y_pred, (224, 224))

    # Calculate perceptual loss using the pre-loaded loss_model
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))

def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block for feature enhancement."""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([input_tensor, se])

def residual_block(x):
    """A small ResNet-like block."""
    res = Conv2D(64, (3, 3), padding="same")(x)
    res = BatchNormalization()(res)
    res = PReLU(shared_axes=[1, 2])(res)
    res = Conv2D(64, (3, 3), padding="same")(res)
    res = BatchNormalization()(res)
    return Add()([x, res])  # Skip connection

def srresnet(num_res_blocks: int = 16):
    """
    Creates SRResNet model.

    Parameters
    ----------
    num_res_blocks: int
        Number of residual blocks in the model
        Default=16

    Returns
    -------
        SRResNet Model object.
    """
    def PReLU_activation(name):
        return PReLU(Constant(value=0.25), shared_axes=[1,2], name=name)

    def residual_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_1")(layer_input)
        d = PReLU_activation(f"prelu_res_{block_number}")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_1")(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_2")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, scale, i):
      """
      """
      u = Conv2D(256, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
      # Wrap depth_to_space in a Lambda layer
      u = Lambda(lambda x: depth_to_space(x, 2), name=f"pix_shuf_{i}")(u)
      return PReLU_activation(name=f"prelu_up_{i}")(u)

    # ==================
    # Model Construction
    # ==================

    lr_image = Input(shape=(None, None, 3))
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same', name="Conv_ip")(lr_image)
    c1 = PReLU_activation(name="prelu_ip")(c1)

    r = residual_block(c1, 64, 0)
    for i in range(1,num_res_blocks):
      r = residual_block(r, 64, i)

    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', name="conv_out")(r)
    c2 = BatchNormalization(momentum=0.8, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])

    u1 = upsample_block(c2, 2, 1)
    u2 = upsample_block(u1, 2, 2)

    c3 = Conv2D(3, kernel_size=9, strides=1, padding='same', activation="sigmoid", name="conv_final")(u2)

    return Model(lr_image, c3, name="SRResNet")

def edsr(num_filters: int = 64, num_res_blocks: int = 16):
    """
    Creates an EDSR model.

    Parameters
    ----------
    num_filters: int
        Number of filters per convolution layer.
        Default=64

    num_res_blocks: int
        Number of residual blocks in the model
        Default=16

    Returns
    -------
        EDSR Model object.
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle = lambda x: depth_to_space(x, 2)

    def residual_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', name=f"conv_res_{block_number}_1")(layer_input)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, i) :
        u = Conv2D(num_filters*4, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
        u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u)
        return u

    # ==================
    # Model Construction
    # ==================

    x_in = Input(shape=(None, None, 3), name="LR Batch")
    x = Lambda(normalize, name="normalize_input")(x_in)

    x = r = Conv2D(num_filters, 3, padding='same', name="Conv_ip")(x)
    for i in range(num_res_blocks):
        r = residual_block(r, num_filters, i)

    c2 = Conv2D(num_filters, 3, padding='same', name="conv_out")(r)
    c2 = Add(name="add_out")([x, c2])

    u1 = upsample_block(c2, 1)
    u2 = upsample_block(u1, 2)
    c3 = Conv2D(3, 3, padding='same', name="conv_final")(u2)

    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(x_in, x_out, name="EDSR")

def calculate_psnr(firstImage, secondImage):
   # Compute the difference between corresponding pixels
   diff = np.subtract(firstImage, secondImage)
   # Get the square of the difference
   squared_diff = np.square(diff)

   # Compute the mean squared error
   mse = np.mean(squared_diff)

   # Compute the PSNR
   max_pixel = 255
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)

   return psnr

def ssim_loss(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
