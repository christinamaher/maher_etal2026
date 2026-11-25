import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from Neural_Analysis import run_permutation_regression_by_group
x=pd.read_csv(Path(__file__).with_name('simulated_neural_data.csv'))
x=x[x.time==0]
print(run_permutation_regression_by_group(x,group_col='electrode',subject_col='subject',formula='power ~ EV + C(rd) + C(chosen_shape) + C(chosen_color)',regressor_to_permute='EV',n_permutations=100,random_state=42))
