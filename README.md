# Motion Of The Ocean - Deep Learning Project 

Use reinforcement learning to train a simulated humanoid to imitate a variety of motion skills from motioncapture data. Based off of [Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html)


## Dependencies
``pip3 install pybullet --upgrade --user``

``pip3 install tensorflow==2.3.1``

``pip install tensorflow-probability``

``pip install gym``

``OpenGL >= 3.2``

``pip install mpi4py`` (NOTE: You must install ``mpich`` before installing mpi4py. On Macs, you can achive this by running: 
```
brew install mpich
```

## arg_files
A number of argument files are already provided in `args/` for the different skills. 
`train_[something]_args.txt` files are setup for `train_model.py` to train a policy and 
`run_[something]_args.txt` files are setup for `run_visualizer.py` to run the corresponding 
policy located in `Saved_Models/`. Make sure that the reference motion `--motion_file` 
corresponds to the motion that your policy was trained for, otherwise the policy will not run properly.

## Training Models
To train a policy, use `train_model.py` by specifying an argument file and the number of worker processes.
For example,
```
python3 train_model.py --arg_file train_humanoid3d_walk_args.txt
```
will train a policy to walk using 4 workers. It typically takes about 60 millions samples 
to train one policy, which can take a day when training with 16 workers. 

## Running Visualizer on Our Models:
You can visualize how the model performs by running the `run_visualizer.py` file.
For example:

``` 
python3 run_visualizer.py --arg_file run_humanoid3d_walk_args.txt
```
will run the visualizer for the model in `Saved_Models/` that was trained against the walking motion capture data.


## Visualizer Interface 
- ctrl + click and drag will pan the camera
- left click and drag will apply a force on the character at a particular location
- scrollwheel will zoom in/out
- pressing space will pause/resume the simulation

## Team Members
Lucas Schroeder

Peter Zubiago 

Chris Luke

## References 
https://dl.acm.org/doi/10.1145/3359566.3360072 

https://medium.com/@rudraalabs/deep-mimic-with-bvh-data-2ca367cea418

https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8?fbclid=IwAR25_VSFYADHIUjaOZrqz7rZBgvwW11gqk77NMdnd9oHwBf_LCfdapMoSb4

