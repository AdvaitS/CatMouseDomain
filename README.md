# CatMouseDomain
Dependencies: Python3, numpy, matplotlib

Execution instructions -
To view / play the game, run main.py and the outputs will be generated on the console.
If you are playing the interactive version, you will be prompted for inputs before each turn of your agent.

Performance Evaluation - 
To evaluate the performance of the program, the following lines need to be commented from main.py file
line 6 - 21
line 130 - 137 
Once the lines are commented out, execute the file performance_evaluation.py.
To change the grid size while evaluating the performance, dm.SIZE (line 11) can be changed in performance_evaluation.py file.

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
