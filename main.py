import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
from absl import logging

logging.info("Initial Log Message")

def run_notebook(notebook_path, output_path=None, timeout=36000):
    """
    Executes a Jupyter Notebook and optionally saves the output.
    :param notebook_path: Path to the notebook file to execute.
    :param output_path: Path to save the executed notebook (optional).
    :param timeout: Execution timeout for each notebook cell (default: 600 seconds).
    """
    ## Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
    ## Set up the notebook executor make sure it is the correct kernel name
    executor = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    try:
        ## Execute the notebook
        executor.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})

        # #Save the executed notebook if an output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                nbformat.write(notebook, output_file)       
        print(f"Executed notebook: {notebook_path}")
    except Exception as e:
        ## catch errors
        print(f"Failed to execute {notebook_path}: {e}")

def folder_safe(directory_path):
    """ Check if the directory exists
        and if not then create it
    """
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def main():
    """ main script execution
    """
    ## Define the paths to the specific notebooks
    ## Should be set to A and B for official running
    ## Can be changed for development test
    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    # Construct the correct path
    metrics_path = os.path.join(current_directory, "metrics")
    folder_safe(metrics_path)
    notebook_track1_base_path   = "/content/drive/MyDrive/AMLS2/Task_Single_Base.ipynb"
    notebook_track1_hyper_path  = "/content/drive/MyDrive/AMLS2/Task_Multi_Hyper.ipynb"
    notebook_track1_tune_path   = "/content/drive/MyDrive/AMLS2/Task_Single_Tune.ipynb"
    notebook_track1_edsr_path   = "/content/drive/MyDrive/AMLS2/Task_Single_EDSR.ipynb"
    notebook_track1_plus_path   = "/content/drive/MyDrive/AMLS2/Task_Single_Plus.ipynb"
    notebook_track2_tune_path   = "/content/drive/MyDrive/AMLS2/Track2_Tune.ipynb"

    ## Define where to save the executed notebooks (optional)
    executed_track1_base_path   = "/content/drive/MyDrive/AMLS2/executed_notebook_Task_Single_Base.ipynb"
    executed_track1_hyper_path  = "/content/drive/MyDrive/AMLS2/executed_notebook_Task_Multi_Hyper.ipynb"
    executed_track1_tune_path   = "/content/drive/MyDrive/AMLS2/executed_notebook_Task_Single_Tune.ipynb"
    executed_track1_edsr_path   = "/content/drive/MyDrive/AMLS2/executed_notebook_Task_Single_EDSR.ipynb"
    executed_track1_plus_path   = "/content/drive/MyDrive/AMLS2/executed_notebook_Task_Single_Plus.ipynb"
    executed_track2_tune_path   = "/content/drive/MyDrive/AMLS2/executed_notebook_Track2_Tune.ipynb"

    ## Execute all notebooks
    run_notebook(notebook_track1_base_path, executed_track1_base_path)
    run_notebook(notebook_track1_hyper_path, executed_track1_hyper_path)
    run_notebook(notebook_track1_tune_path, executed_track1_tune_path)
    run_notebook(notebook_track1_edsr_path, executed_track1_edsr_path)
    run_notebook(notebook_track1_plus_path, executed_track1_plus_path)
    run_notebook(notebook_track2_tune_path, executed_track2_tune_path)       


if __name__ == "__main__":
    main()
