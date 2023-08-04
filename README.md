# PINNEik Using Jax
This repository contains the Jax PINNEik Codes, which were used for various velocity models and supervised learning. The codes, images, weights, and inputs are organized into their respective folders.

### Code Structure
The `codes` folder contains 6 IPython Notebook scripts, each corresponding to different velocity models, along with one notebook on supervised learning. Each notebook begins with a table of contents summarizing its contents, which can be used for easy navigation.

### System and Installation Requirements
To run the notebook scripts, ensure that you have Jax for CUDA installed. This allows the scripts to utilize the GPU. If Jax[CUDA] is not installed, you may encounter errors when importing the necessary libraries.

### GPU Setting & Code Reproducibility
The notebook scripts have been optimized to run on a GPU. The "GPU Setting & Code Reproducibility" cell checks for the availability of a GPU and prints either "gpu" or "cpu" accordingly. If the notebook is running on a CPU, a warning message will be displayed indicating that no GPU was found and the code will fall back to CPU.
If a GPU is detected, the cell will output: "This Notebook is currently running on a: gpu," along with the GPU details. 
If running on GPU, before proceeding to run other cells, please ensure that the following line of code shown below are applicable to your case 
>>>  GPU = 1

>>> os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU

>>> output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')

>>> gpu_info = output.strip().split('\n')

>>> print(gpu_info[GPU])

This assigns a specific GPU (in this case GPU 1) to the notebook and prints out the details of the GPU assigned. Kindly ensure this is applicable to your system. For instance, attempting to assign “GPU=1” to a system that only has one GPU (indexed as 0) would generate an error. 
Consequently, this should be assigned as is applicable to your circumstance. Otherwise, it may be best to comment out the above lines of code.
Please note that if no GPU is detected and Jax_XLA falls back to CPU. The notebook may still run (If Jax for cpu is installed), but the training time will be significantly affected.
However, kindly ensure you do not comment out the line shown below:

>>> random_key = random.PRNGKey(0)

This defines a jax.random key used at different points within the code. Commenting out this line alongside the GPU assignment lines would generate the error “random_key not defined” at certain cells below.

### File Paths
The default file paths within the notebooks should be edited to reflect the actual file paths on your system. Failure to do so may result in a "File path not found" error. Please update the relevant file paths to ensure proper execution.

### Marmousi Model and Elliptical Anisotropic Models
The Marmousi model and elliptical anisotropic models contain functions for random parameter initializations. However, these functions have been commented out, as the scripts were originally written to use pre-trained weights. If you want to utilize the random parameter initializations, simply uncomment these functions within the respective scripts.

### Efficient Notebook Rendering
To render the Jupyter Notebook pages reliably without downloading, copy and paste the URL for this repository at: https://nbviewer.jupyter.org/ The files will be rendered for quick view.

### Reference
This repo is based on the original work PINNEik by Waheed et al. (2021). All materials were retrieved from the repo: https://github.com/umairbinwaheed/PINNeikonal/ 
Link to the original paper: https://doi.org/10.1016/j.cageo.2021.104833

Please feel free to explore the provided code and customize it according to your requirements. If you have any questions or encounter any issues, don't hesitate to reach out for assistance. 


