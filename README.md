This repository is associated with the paper: *Shallow neural networks trained to detect collisions recover features of visual loom-selective neurons*, where we built a neural network that can identify whether or not the incoming visual signals indicate an object on a collision course. The model exhibit a large range of properties, including structures, response curves and tunings, that resemble the loom-sensitive LPLC2 neurons in the fly brain. 

## The folders are organized as follows:

**stimulus_core**: core files to generate stimuli for training and testing.

**stimulus_generation**: files to perform the stimuli generation processes.

**models_core**: core files of the models and trainings.

**models_training**: files to perform the training and testing processes.

**analysis**: files and notebooks to perform data/model analysis and figure generations.

**helper**: files that contain all the helper functions.

**results**: main results, figures and videos are saved here.

**env**: environment.

## How to use

0) The environment can be set up using `py37_dev.yml` in the folder **env**. 

1) Need at least 300 GB of space to save the stimuli and training results. The whole process takes days or even weeks to run depending on your computing power. 

2) The letter M indicates the number of units in the model, and in our paper, it takes values 1, 2, 4, 8, 16, 32, 64, 128, 192, 256. 

3) Use `samples_generation_multi_units_run_smallM.py` and `samples_generation_multi_units_run_largeM.py` in the folder **stimulus_generation** to generate the stimuli for training and testing. You should use more than 30 cores to do this.

4) Use `train_multiple_units_M{M}.py` in the folder **models_training** to train and test the model, where {M} indicates 1, 2, 4, 8, 16, 32, 64, 128, 192, 256. You should use more than 30 cores to do this.

5) Use `model_clustering.ipynb` in **analysis** to do the model clustering. You need to run the whole notebook for very M, and so that you can check the clustering by eyes through some figures generated. This sets the foundation for Figures 5-10.

6) Use `get_samples_for_movie_single_unit.py` and `get_samples_for_movie_multi_units.py` in **analysis** to generate high resolution samples for plotting of both movies and figures. This is for Figure 3, Figure 8 and its supplementations, and videos.

7) Use `movie_3d.py` and `movie_3d.ipynb` in **analysis** to generate 3d rendering samples. This is for Figure 3, and videos.

8) Use `calculate_response_vs_distance_vs_angles.py` in **analysis** to calculate the responses of the models to different types of stimuli. This is for Figure 7.

9) Use `calculate_incoming_angles.py` in **analysis** to calculate the incoming angles of hit stimuli. This is for Figure 7.

10) Use `samples_generation_grid.py` in **stimulus_generation** and `get_grid_response.py`  in **analysis** to generate grid samples and grid responses. This is for Figure 7.

11) Use `sparse_dense_response.py` in **analysis** to calculate active units in models. This is for Figure 8.

12) Use `get_probability_of_hit.py` in **analysis** to calculate probability of hit. This is for Figure 8.

14) Use `get_Klapoetke_stimuli.py` in **stimulus_core**, `samples_generation_linear_law.py` in **stimulus_generation** and `replication.py` in **analysis** to reproduce experimental data. All experimental data are from Card's lab. This is for Figure 10.

15) Use `tuning_curve.py` in **stimulus_generation** to generate data for HRC tuning curves. This is for supplementation figure of Figure 3.

16) Use `get_HRC_output_all.py` in **analysis** to collect all the HRC outputs for different data types, respectively. This is for supplementation figure of Figure 3.

17) Use `plot_all_movies.ipynb` in **analysis** to get the movies.

18) Use `plot_all_main_figures.ipynb` in **analysis** to get the figures.

## Contributions

Stimuli generation, figures and videos: Baohua Zhou

Model training: Zifan Li, Baohua Zhou







