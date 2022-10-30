# NIFM (Neural Integration-free Flow Maps)
[Saroj Sahoo]() [Matthew Berger](https://matthewberger.github.io/)

This paper contains the official code for the paper "Integration-free Learning of Flow Maps".

## Requirements
The relevant libraries required to run this project are included in the `requirements.txt` file.

## Training NIFM
Training NIFM is a two-stage process, in the first stage we fit to the vector field. Our method expects the 2D and 3D time-varying vector field data to be stored in a specific way, which we list below:-

### 2D Time-Varying Vector Field Data
Assuming that the entire 2D dataset can fit into memory, our method expect a single `field.npy` file containing the entire 2D time-varying dataset with the shape `[2,t,x,y]` where `t`,`x` and `y` corresponds to the time, (x, y) spatial dimensions respectively. Addtionally, our method expects a `metadata.json` file which contains the metadata for the dataset placed in the same directory where the 2D time-varying field is stored. Below is an example of the `metadata.json` for the Fluid Simulation dataset.

```
{
    "dataset_name": "fluid_sim_re_16",
    "res": [1001, 512, 512],
    "ext": [[0.0,10.0],
            [0.0,1.0],
            [0.0,1.0]],
    "toroidal": true
}
```

### 3D Time-Varying Vector Field Data
Since, most 3D time-varying data can get really large, the luxury of the storing them in memory is no longer feasible. So, our method expects files named `field_0.npy`, `field_1.npy`, ..., `field_100.npy`, where each file corresponds to a single slice and has a shape of `[3,x,y,z]`. Similar to 2D dataset, a `metadata.json` for the entire 3D dataset needs to placed in the same directory.

### Fitting to Vectors
After the data prep is done, fitting to the vectors can be done by running the `optimize_vectors.py` script. Below is an example:
```
python optimize_vectors.py --data_dir=path_to_2d_vector_field_data --n_dim=2 --experiment=name_of_the_experiment
```
The `--experiment=name_of_the_experiment` argument will create a `Experiments\name_of_the_experiment` folder within the Project Directory where all the relevant files will be written out for this experiment. By default, we use a compression level of 10 for 2D time-varying datasets, however this can be changed using the `--compression_ratio` argument. Similarly, other network related arguments can be explored. After the training is done, a file `Experiments\name_of_the_experiment\net_info.json` containing the network information will written out along with the saved model weights in the same directory.


### Fitting to Flow maps
After the first stage, in second stage the model needs the network information from the previous stage and thus will look for it in the subfolder with the appropriate experiment name. Thus, the `--experiment` argument must be provided with the same experiment name as in the 1st stage of training. For examples:-
```
os.system("python optimize_flow_map.py --data_dir=path_to_2d_vector_field_data --n_dim=2 --experiment=name_of_the_experiment")
```
Additionally, the user can set the `min_tau` and `max_tau` (in grid units) to specify the range of time-spans the networks has to learn through the self-consistency criterion. After training is finished the model weights and the network information file `net_flow_info.json` will be saved out in the corresponding folder.

### Quantitative and Qualitative evaluation
The evaluation of the network is straightforward and can be done using the following scripts :-

```python quantitative_eval.py --data_dir=path_to_2d_vector_field_data --experiment=name_of_the_experiment```
The above script will evaluate our method quantitatively for varying start times and time-spans.

And, the following script can be used to compute the ground truth FTLE as well as FTLE using our method.

```python ftle_eval.py --data_dir=path_to_2d_vector_field_data --experiment=name_of_the_experiment --start_time=0 --tau=1 --grid_steps=16```

The `--grid_steps` arguments controls how many steps in grid units the models takes to compose the flow map of time-span set by `--tau` argument.

