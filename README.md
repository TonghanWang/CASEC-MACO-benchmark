# CASEC: Context-Aware Sparse Deep Coordination Graphs

# MACO: Multi-Agent Coordination benchmark

This codebase is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) and contains the implementation
of the Multi-Agent COordination (MACO) benchmark and CASEC algorithm.

## Run an experiment 

Tasks in the MACO benchmark can be found in `src/envs`. To run experiments on the MACO benchmark:

```shell
python src/main.py --config=casec --env-config=hallway with threshold=0.5 t_max=1050000 use_action_repr=False construction_delta_var=True delta_var_loss=True independent_p_q=True
```

To run experiments on the SMAC benchmark:

```shell
python src/main.py --config=casec --env-config=sc2 with env_args.map_name=MMM2 use_action_repr=True delta_var_loss=True construction_delta_var=True threshold=0.3 t_max=2005000 independent_p_q=False
```

There are four methods for building sparse graphs:
* `construction_delta_abs`: Using the maximum utility difference (Eq. 5 in the paper)
* `construction_q_var`: Using the variance of payoff functions (Eq. 6 in the paper)
* `construction_delta_var`: Using the variance of utility difference functions (Eq. 7 in the paper)
* `construction_attention`: Using the attentional observation-based approach (Eq. 9 in the paper)

By default, they are set to `False`. Setting `True` for one of them would use the corresponding method to construct sparse graphs. Setting `full_graph` or `random_graph` to `True` can test complete or random coordination graphs, respectively.

There are three losses for learning sparse topologies (Eq. 8 in the paper):
* `l1_loss`: Using ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathrm{sparse}}^{|\delta|}) 
* `q_var_loss`: Using ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathrm{sparse}}^{q_{\mathrm{var}}})
* `delta_var_loss`: Using ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathrm{sparse}}^{\delta_{\mathrm{var}}})

By default, they are set to `False`. Setting `True` for one of them would use the corresponding loss.

CASEC uses `construction_delta_var` and `delta_var_loss`. 
The config files act as defaults for an algorithm or environment. 
They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.
All results will be stored in the `Results` folder.
The previous config files used for the SMAC Beta have the suffix `_beta`.

## Installation instructions

Build the Dockerfile using:
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder named *models*. The directory corresponding to each run will contain models saved throughout the training process, each of which is named by the number of timesteps passed since the learning process starts.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
