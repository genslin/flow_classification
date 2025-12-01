# Description
Machine learning classification of two-phase annular flow images using the Texas A&M Dataset 
for Purdue NUCL57500 project

# Quickstart:
The interactive model builder is handled through the command line interface. The project is built on uv.
## Installing uv
See: [Astral UV Installation Instructions](https://docs.astral.sh/uv/getting-started/installation/)
## Installing dependancies
Once uv is installed on your computer simply run 
```bash
uv sync --extra cpu 
```
to download the needed dependencies, if you have gpu on your machine then you can run:
```bash
uv sync --extra cu128
```
## Running the program
```bash
uv run main.py 
```
followed by one of the subcommands [train, test, plot, autotrain] and available arguments you are trying to manipulate starts the script.
## Help
```bash
uv run main.py --help
```
give you the optional parameters you can manipulate 
## Train Subcommand
```bash
uv run main.py train
```
Used to create/load and train models
### Creating a new model
```bash
uv run main.py train --model-name "Model Name" 
```
Creates a new model for you to work with
### Loading an existing model
```bash
uv run main.py train --model-name "Model Name" --load-existing
```
Loads an existing model
### Quick run
```bash
uv run main.py train
```
Loads the default_resnet18, training here will be overwritten the next time the default_resnet18 model is used so make sure to make a named model if you want to your weights to be permanently saved
#### Session Logging
All of the work that you do is saved in the applicable model_name within saved_models/, checkout the log file for the interesting information from what you just did
## Test Subcommand
```bash
uv run main.py test --model-name "Model Name" 
```
Tests the model against the withheld data, should only be done when you're completely done training the model
## Plot Subcommand
```bash
uv run main.py plot --model-name "Model Name" 
```
Plots the performance data accumulated from training a model
## Autotrain Subcommand
```bash
uv run main.py autotrain --model-name "Model Name" 
```
Creates and trains a model using the default settings for 10 epochs in head only then 10 epochs with layer4 and head training. Will rewrite the previous model if used on an existing model-name (does not stack)