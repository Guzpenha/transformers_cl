import argparse
from IPython import embed
import pandas as pd
import os

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--slurm_file",
                    default=None,
                    type=str,
                    required=True,
                    help="")

parser.add_argument("--num_iter",
                    default=18,
                    type=int,
                    required=True,
                    help="")

parser.add_argument("--run",
                    default=None,
                    type=str,
                    required=True,
                    help="")


args = parser.parse_args()

os.system("rm res")
os.system("cat "+args.slurm_file+" | grep -Po 'map = \\K\\d+.\\d+' > res")
print("cat "+args.slurm_file+" | grep -Po 'map = \\K\\d+.\\d+' > res")
sets = ['standard_training', 'geom_progression', 'step', 'linear', 'root_2', 'root_5', 'root_10', 'root_20']
set_idx = 0
df = []
with open("res", 'r') as f:
	results = [r.strip() for r in f.readlines()]
	final_res = []
	for i,v in enumerate(results):
		if (i+1) % args.num_iter != 0:
			df.append([str((i+1) % args.num_iter) , v ,sets[set_idx]])
			# print(str((i+1) % args.num_iter) + "\t" + v + "\t" +sets[set_idx])
		else :
			final_res.append(v)
			set_idx+=1
	print("Final Results : ")
	print('\t'.join(sets))
	print('\t'.join(final_res))
	df = pd.DataFrame(df, columns = ["iter", "map", "curriculum"])
	print(df)
	print(df.groupby("curriculum")["map"].apply(max))
	df.to_csv("eval_pacing_func_during_training_bert_"+args.run+".csv", index=False)