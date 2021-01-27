# InterpretMutations
A formalism for understanding the contributions of different factors to the mutational effects predictions

C. K. Sruthi, M. K. Prakash, Disentangling the contribution of each descriptive characteristic of every single mutation to its functional effects (2020).

Older version of the manuscript is available at: https://www.biorxiv.org/content/10.1101/867812v2


INPUTS.
The input features, reported fitness, scaled fitness and predicted fitness are given for each protein in the files:
blact.csv - Beta-lactamase 
aph3.csv  - APH(3')-II 
hsp90.csv - Hsp90
mapk1.csv - MAPK1
ube2i.csv - UBE2I
tpk1.csv  - TPK1
bgl3.csv  - Bgl3
lgk.csv   - LGK

HYPERPARAMETERS.
The script to optimize hyperparameters is OptimizeparamsPred.py.
The command "python OptimizeparamsPred.py -h" will print the required inputs to run the script.
An example to run this script is:
python OptimizeparamsPred.py -i aph3.csv -l 0.01 -f 14

The tuned parameters obtained by running OptimizeparamsPred.py can be fed to shapanalysis.py to develop XGBoost models with the optimized parameters and to perform the SHAP analysis.


SHAP analysis.
The script to perform the SHAP analysis is shapanalysis.py.
The command "python shapanalysis.py -h" will print the required inputs to run the script.
An example to run this script is:
python shapanalysis.py -i aph3.csv -o aph3fit -f 14 -t 0.25 -l 0.01 -n 1253 -d 7 -w 15 -g 0 -s 0.8 -c 0.4 -a 0.01

LEAVE ONE FEATURE OUT analysis.

Hyperparameter tuning as well as the SHAP analysis can be done removing a single feature at time using the scripts OptimizeparamsPred.py and shapanalysis.py with the minor modifications described below. 

1. Add one more argument to pass the name of the feature to be removed by including this line

parser.add_argument('-r','--remove', dest= "remove_col", help="The column to be removed when generating the model",type=str,required=True)

after line number 22 in "OptimizedparamPred.py" and after line number 34 in shapanalysis.py

2. Remove the feature specified by this argument by

X = X.drop(columns=[options.remove_col])

after line number 34 in "OptimizedparamPred.py" and after line number 46 in shapanalysis.py

Then the scripts can be executed using commands like

REPEATING HYPERPARAMETER optimisation.
python OptimizeparamsPred.py -i blact.csv -l 0.01 -f 15 -r Conservation 
This will tune hyperparameters removing the feature Conservation from inputs. The input to be provided following the argument -r is the feature name as it appears in the input file, here blact.csv.

REPEATING SHAP analysis.
Similarly the script shapanalysis.py can be executed with the extra argument added.
For example:
python shapanalysis.py -n 1138 -l 0.01 -d 25 -w 20 -g 0 -s 0.9 -c 0.9 -a 0 -f 21 -o out -i blact.csv -t 0.25 -r Conservation

Similar calculations can be done for all features by passing suitable arguments.
