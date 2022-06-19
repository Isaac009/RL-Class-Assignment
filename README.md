# RL-Class-Assignment
Reinforcement Learning Homework 2
This repository contains the project source codes used to train an agent to play two video games.
The baseline source codes were borrowed from https://github.com/deepanshut041/Reinforcement-Learning

# Software requirements
Please create a python virtual environment and activate it and then clone this repository using this command 
`git clone https://github.com/Isaac009/RL-Class-Assignment.git`

After that, run the following command to install some dependencies

`pip install -r requirements.txt`


# ROM
You will need roms to be able to either train or test the models.
Once you have them, put them in any directory, let say `Homework2/roms/`

Then run the following command:
`python3 -m retro.import roms/`

This command will import all available valid roms. 
Now you are ready to either train your model from scrach or load the trained model from the checkpoints and test them to see the smart agent.

