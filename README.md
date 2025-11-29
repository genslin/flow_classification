# Description
Machine learning classification of two-phase annular flow images using the Texas A&M Dataset 
for Purdue NUCL57500 project

# Quickstart:
The interactive model builder is handled through the command line interface. The project is built on uv.
## Installing uv
See: [text](https://docs.astral.sh/uv/getting-started/installation/)
## Installing dependancies
Once uv is installed on your computer simply run 
>>>uv sync --extra cpu 
to download the needed dependencies, if you have gpu on your machine then you can run:
>>>uv sync --extra cu128
## Running the program
>>>uv run main.py 
followed by whatever arguments you are trying to manipulate starts the script.
## Help
>>>uv run main.py --help
give you the optional parameters you can manipulate 
## Creating a new model
>>> uv run main.py --model-name "Model Name" 
Creates a new model for you to work with
## Loading an existing model
>>> uv run main.py --model-name "Model Name" --load-existing
Loads an existing model
## Quick run
>>> uv run main.py 
Loads the default_resnet18, training here will be overwritten the next time the default_resnet18 model is used so make sure to make a named model if you want to your weights to be permanently saved
### Session Logging
All of the work that you do is saved in the applicable model_name within saved_models/, checkout the log file for the interesting information from what you just did