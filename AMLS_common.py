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

#################################################### LIBRARY IMPORTS ##############################
## standard python libraries
import datetime
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
## set up tensorflow
import tensorflow as tf
## MedMNIST specific libraries loading all relevant items (updated 07122024)
##import medmnist
##from medmnist import INFO ##, info
## sklearn to allow analysis of hyperparameter choices
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

#################################################### DATA LOADING ##############################



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
    if type_flag in ['accuracy','loss']:
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
