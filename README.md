# Tuning On-Policy Policy Gradient Algorithms in JAX

## Overview

This repository builds upon the work of Matthias Lehmann on policy gradients implemented in JAX, with a focus on hyperparameter tuning for various on-policy policy gradient algorithms. The main objective is to identify optimized configurations for these algorithms and save them in the `hyperparams` folder for reuse. The repository leverages a library version of [PolicyGradientsJax](https://github.com/pre63/policy-gradients-jax) to run the algorithms and streamline hyperparameter optimization.

## Running Hyperparameter Tuning

Hyperparameter tuning for specific algorithms can be executed using a single `make` command. For example, tuning the Trust Region Policy Optimization (TRPO) algorithm for the `Humanoid-v5` environment involves running the command:

```bash
make train model=trpo env=Humanoid-v5
```

This command automates the training and tuning process, ensuring efficient experimentation and reproducibility.

## Running Algorithms Locally

The repository also supports running algorithms locally for evaluation purposes. Executing a model with a pre-configured setup can be achieved with commands such as:

```bash
make evaluate model=trpo env=Humanoid-v5
```

This command evaluates the chosen algorithm on a specific environment. These streamlined commands ensure a seamless user experience when testing or deploying models.

## Supported Algorithms

A range of on-policy policy gradient algorithms are included, each with a strong foundation in reinforcement learning research. These algorithms include:

- **REINFORCE**: The foundational policy gradient algorithm that provides a basis for on-policy methods.  
- **Advantage Actor-Critic (A2C)**: An improvement over REINFORCE, leveraging advantage estimation for efficiency.  
- **Trust Region Policy Optimization (TRPO)**: Introduces a trust-region constraint for stable policy updates.  
- **Proximal Policy Optimization (PPO)**: Enhances TRPO with clipped objectives for scalability and practicality.  
- **V-MPO**: A probabilistic approach to policy optimization designed for efficiency in large-scale environments.

Each algorithm is implemented with a focus on performance and tunability, backed by foundational research and publications.

## Addressing Known Issues

The repository addresses common compatibility issues, such as those encountered with Mujoco on recent macOS versions. OpenGL-related problems can be resolved by updating the library configuration in the Mujoco package. This involves replacing the line:

```python
_CGL = ctypes.CDLL('/System/Library/OpenGL.framework/OpenGL')
```

with:

```python
_CGL = ctypes.CDLL('/opt/X11/lib/libGL.dylib')
```

Alternatively, the process can be automated using the provided `sed` command:

```bash
sed -i.bak "s|_CGL = ctypes.CDLL('/System/Library/OpenGL.framework/OpenGL')|_CGL = ctypes.CDLL('/opt/X11/lib/libGL.dylib')|" .venv/lib/python3.11/site-packages/mujoco/cgl/cgl.py
```

These fixes ensure smooth execution of Mujoco environments, which are integral to reinforcement learning experiments.

## Conclusion

By combining robust algorithm implementations, automated tuning processes, and compatibility solutions, this repository serves as a comprehensive platform for experimenting with on-policy policy gradient algorithms in JAX. Researchers and practitioners can use this resource to explore, optimize, and deploy cutting-edge reinforcement learning models.
