# CatMouseDomain
Dependencies: Python3, numpy, matplotlib

Execution instructions -
To view / play the game, run main.py and the outputs will be generated on the console.
If you are playing the interactive version, you will be prompted for inputs before each turn of your agent.

Performance Evaluation - 
To evaluate the performance of the program, the following lines need to be commented from main.py file
line 6 - 25
line 130 - 137 
Once the lines are commented out, execute the file performance_evaluation.py.
To change the grid size while evaluating the performance, dm.SIZE (line 11) can be changed in performance_evaluation.py file.

Training your own Neural Network -
To train a new neural network, you can run the file nn.py with your own values for learning rate (Line 84) and other parameters. Make sure lines 6-25 and 144-151 in main.py are commented out before running nn.py
This file generates training data by playing 10 games and collecting all the grids generated in them, and testing data by playing 2 games and collecting them.

domain_mcst.py -
This file contains all the helper functions required to implement the Tree search as well as Tree+NN implementation. 

mcst_nn.py - 
This file contains functions to return the pre-trained models (lin_lin_0006, lin_lin_0001, lin_0005, conv_lin).

Following functions were referred from the code examples shown in class and is implemented in domain_mcst.py. 
link : https://colab.research.google.com/drive/1JuNdI_zcT35MWSY4-h_2ZgH7IBe2TRYd?usp=sharing#scrollTo=3pFmEP4A--e9
1. Rollout
2. uct
3. Exploit
4. N_Values
5. Q_Values
6. Children
7. Is_Leaf
8. Children_of
9. Score

The neural network architectures have been refered from the class colab notebooks and PyTorch Documentation:
https://colab.research.google.com/drive/1zHTZyqZoOq4Hqx5oGl1rPFVnanB_IBUF?usp=sharing#scrollTo=bhEjp3zUoTUN
https://colab.research.google.com/drive/1QF8IJHlZ597esIU-vmW7u9KARhyXIjOY?usp=sharing#scrollTo=txiloxBPJnVM

The remaining .png, .jpg files are performance evaluation results.
The params_<>.txt files have the weights and biases saved for the different trained models.
