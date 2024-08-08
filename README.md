# DIAMBRA Arena Project

This repository contains code and configurations for running a DIAMBRA Arena environment, leveraging Stable-Baselines3 for agent training and evaluation for the game tekken tag tournament.

## Installation

To install DIAMBRA Arena along with Stable-Baselines3, use the following command:

    pip3 install diambra-arena[stable-baselines3]

For more detailed instructions and documentation, please refer to the official DIAMBRA documentation: [docs.diambra.com](https://docs.diambra.com)

## Running the Environment

To run the environment, use the following command:

```
diambra run -r /absolute/path/to/roms/folder/ -l -g --engine.sound python submission.py --cfgFile /path/to/cfg/file --modelPath /path/to/model
```

- Replace `/absolute/path/to/roms/folder/` with the path to your ROMs folder.
- Replace `/path/to/cfg/file` with the path to your configuration file.
- Replace `/path/to/model` with the path to your trained model.

## Repository Structure

- **submission.py**: The main script to run the environment.
- **cfg**: Configuration files for the DIAMBRA Arena.
- **models**: Pre-trained models or models generated during training.

## Acknowledgements

- [DIAMBRA Arena](https://diambra.com) - For providing the platform for competitive AI environments.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - For reinforcement learning algorithms.

---

Happy coding! 🚀
