from IPython import embed
import pandas as pd
from functools import reduce
import numpy as np
import scipy.stats
from functools import reduce


task = "mantis_10"

for seed in ['111']:
	df = pd.read_csv(
		"data/preds_run_cl__seed_"+str(seed),
		names=["preds_run_"+str(seed)])
	difs_df = []
	for i in range(len(df)):
		if i % 2 != 0:
			difs_df.append(df.values[i-1][0]-df.values[i][0])
	difs_df = pd.DataFrame(difs_df,columns = ["preds_dif_run_"+str(seed)])
difs_df.to_csv("data/"+task+"_bert_preds_dif_c_values_3", header=False,index=False)		