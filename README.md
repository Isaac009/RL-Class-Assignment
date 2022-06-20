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

# Testing Args and their default values
- parser.add_argument('--game_title', type=str, default='MortalKombatII-Genesis',
                    help='Game title as our environment Should be exactly the same as the rom you want to use (default: MortalKombatII-Genesis)')
- parser.add_argument('--version', default=1,  type=str, required=True,
                    help='version the integer (default: 1) - checkpoint saved at timestep t')
- parser.add_argument('--mode', default=0,
                    help='mode the integer (default: 1) - training mode 1 else 0 (test mode)')
- parser.add_argument('--video_path', default='videos',  type=str, help='Video records path (default: videos)')
- parser.add_argument('--ckpt_path', default='checkpoints/MortalKombat/ppo_actor_',  type=str,
                    help='Checkpoints path for trained models (default: checkpoints/mortal_kombat/ppo_actor_)')

## Testing models command example
`python test_mortal_kombat_2.py --game_title=StreetOfRange3-Genesis --version 240000 --mode 0 --ckpt_path=checkpoints/StreetOfRange/ppo_actor_`
