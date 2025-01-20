# Tuning On-Policy Policy Gradient Algorithms in JAX

This repository builds upon the work of Matt00n on policy gradients implemented in JAX. The primary focus here is on tuning the hyperparameters of various on-policy policy gradient algorithms and saving the optimized configurations in the `hyperparams` folder. We use a [library version of PolicyGradientsJax](https://github.com/pre63/policy-gradients-jax) to run the algorithms and tune the hyperparameters.

## Running Hyperparameter Tuning

To tune hyperparameters for a specific algorithm, use the corresponding `make` command.

```bash
make train model=trpo env=Humanoid-v5
```

## Running Algorithms Locally

To run an algorithm locally, execute the corresponding Python file through the `make` command.:

```bash
make evaluate model=trpo env=Humanoid-v5
```

## Algorithms

This repository includes the hyperparameters for the following algorithms:

- [REINFORCE (reinforce)](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) 
- [Advantage Actor-Critic (a2c)](https://arxiv.org/abs/1602.01783) 
- [Trust Region Policy Optimization (trpo)](https://arxiv.org/abs/1502.05477) 
- [Proximal Policy Optimization (ppo)](https://arxiv.org/abs/1707.06347) 
- [V-MPO (vmpo)](https://arxiv.org/abs/1909.12238) 


## Known Issues
**OpenGL on recent macOS**  

Replace  
`_CGL = ctypes.CDLL('/System/Library/OpenGL.framework/OpenGL')`  
with  
`_CGL = ctypes.CDLL('/opt/X11/lib/libGL.dylib')  
`  
in  

```
nano .venv/lib/python3.11/site-packages/mujoco/cgl/cgl.py
```
or
```
sed -i.bak "s|_CGL = ctypes.CDLL('/System/Library/OpenGL.framework/OpenGL')|_CGL = ctypes.CDLL('/opt/X11/lib/libGL.dylib')|" .venv/lib/python3.11/site-packages/mujoco/cgl/cgl.py
```

## Citing PolicyGradientsJax

If you use PolicyGradientsJax in your research or find it helpful, please consider citing Matthias Lehmann's [paper](https://arxiv.org/abs/2401.13662):
```bibtex
@article{lehmann2024definitive,
      title={The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations}, 
      author={Matthias Lehmann},
      year={2024},
      eprint={2401.13662},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```