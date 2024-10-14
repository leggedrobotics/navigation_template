# Template for _Navigation_ Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository fork serves as a template for building projects or extensions based on Isaac Lab, specifically for navigation projects. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

Out the box, the code here should allow you to train a basic navigation policy using one forwards-facing stereolabs zed camera, in a basic environment.

The code skeleton here is very lightweight, because all shared navigation components live in [isaac-nav-suite](https://github.com/leggedrobotics/isaac-nav-suite), and this template just assembles them in an example `env_cfg` file in a way that works out the box, and demonstrates how to set up your directory structure. **NOTE:** at present, the necessary navigation components are not yet checked into the main branch of isaac-nav-suite, so to get a working project out the box, you should check out the [dev/kappi/perceptnet](https://github.com/leggedrobotics/isaac-nav-suite/tree/dev/kappi/perceptnet) branch.

Setup Tips:

- You should use the "use this template" button instead of forking this repo, so it doesn't end up public.
- Once you have got the template, you should add it as a git submodule to IsaacLab-Internal
- Remember to run `git submodule update --init --recursive` in IsaacLab-Internal, to automatically get the right version of isaac-nav-suite pulled down (it is a submodule in this template)
- Also remember to symlink the `isaac-nav-suite` modules and your extension module in `IsaacLab-Internal/source/extensions`, like so:
  
![image](https://github.com/user-attachments/assets/d07b24e2-28f7-45b3-b0fc-909a935c5199)
![image](https://github.com/user-attachments/assets/19b5f571-9741-4937-bc79-a41ba36ec40c)

The rest of this README is the same as the general [extension template](https://github.com/isaac-sim/IsaacLabExtensionTemplate) README.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/isaac-sim/IsaacLabExtensionTemplate.git

# Option 2: SSH
git clone git@github.com:isaac-sim/IsaacLabExtensionTemplate.git
```

- Throughout the repository, the name `navigation_template` only serves as an example and we provide a script to rename all the references to it automatically:

```bash
# Enter the repository
cd navigation_template
# Rename all occurrences of navigation_template (in files/directories) to your_fancy_extension_name
python scripts/rename_template.py your_fancy_extension_name
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e exts/navigation_template
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Navigation-NavigationTemplate-PPO-Anymal-D-DEV
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `exts/navigation_template/navigation_template/ui_extension_example.py`. For more information on UI extensions, enable and check out the source code of the `omni.isaac.ui_template` extension and refer to the introduction on [Isaac Sim Workflows 1.2.3. GUI](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html#gui).

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `IsaacLabExtensionTemplate/exts`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source/extensions`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/exts/navigation_template"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
