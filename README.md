***Set Up and Installation***

This Project requires Python, Pytorch and Pygame to run. Regardless of whether you have a NVDIA, AMD GPU or an APU, Python installation is
necessary.

You may download Python here, search for the correct version for you. Ensure you add Python to your PATH's.

https://www.python.org/downloads/

**************************************************************** **Installing Pytorch** ******************************************************************

---------------------------------------For CPU Only and any Non-Nvdia, Non-Highend 7000 Series AMD GPU----------------------------------------

If you lack any compatible GPU's run these commands in terminal

*************************************************************************************
pip install torch torchvision torchaudio
*************************************************************************************

-------------------------------------------------------------------------For Nvdia-------------------------------------------------------------------------
Run these commands in terminal
*************************************************************************************
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
*************************************************************************************

----------------------------------------------------------For AMD Radeon RX 7900 XTX/ XT/ GRE-------------------------------------------------------

List of supported AMD GPU's May be found here

https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html

Pytorch does not run within Windows. You would need to install a Linux distribution, I 
reccomend Ubuntu. You may also Use WSL (Windows Subsystem for Linux)

This may be done here :

https://ubuntu.com/download/desktop#how-to-install-NobleNumbat

Once you have your Linux distribution running, run these commands to install ROCm
within Linux
*************************************************************************************
sudo apt update

wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb

sudo apt install ./amdgpu-install_6.1.60103-1_all.deb

sudo amdgpu-install -y --usecase=graphics,rocm

sudo usermod -a -G render,video $LOGNAME

*************************************************************************************

More Installation Information here:
https://www.amd.com/en/developer/resources/ml-radeon.html

*************************************************************************************


**************************************************************** **Installing Pygame** ******************************************************************
Run these commands in terminal
******************************************************************
pip install pygame
******************************************************************
**************************************************************** **Installing Numpy** ******************************************************************
Run these commands in terminal
******************************************************************
pip install numpy
******************************************************************

***Running the Model***

To run the AI model, simply run the Main file. This should create a popup window that automatically plays tetris. You may close this Window, the game will
continue running in the background. Print statements will appear on console detailing the progress.

***How to manually Play the Game***

To play Tetris run the manualTest file. This should create a popup window, use the *LEFT* and *RIGHT* arrow keys to move the pieces across the board. The *UP*
key changes the orientation the piece. The *DOWN* key can be used to drop the piece faster. Use the *C* key to hold your current piece, you may not do this consecutively.

***How to modify Settings***

To change hyperparameters, you may use the config file. Most notably the episode count can be adjusted in line 12 "NUM_EPISODES" it is currently set to 20000 which is excessive for simply testing the program. I would suggest changing this to 100, as the first 100 runs, are relatively quick, then they begin taking a considerable amount of time. But feel free to change this as necessary.

***Where will the database and model be saved?***

Those files will be saved within the project folder.

***Directory Structure***

Tetris-AI/
│
├── Main.py                 # Main script to run the AI
├── manualTest.py           # Script to play Tetris manually
├── tetrisEnvironment.py    # Environment and gameplay logic
├── dqnAgent.py             # Deep Q-Learning agent
├── config.py               # Configuration and hyperparameter settings
├── requirements.txt        # Dependencies for the project
├── saved_models/           # Directory for saved model weights
└── database/               # Directory for gameplay statistics
