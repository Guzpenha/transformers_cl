import argparse
from IPython import embed
import pandas as pd

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--slurm_file",
                    default=None,
                    type=str,
                    required=True,
                    help="")

args = parser.parse_args()



sets = ["random_batches", "turns", "query_length", "doc_length", "exact_match", "semantic_match", "map_exact_match", "map_semantic_match"]
set_idx = 0
df = []
with open("res", 'r') as f:
	results = [r.strip() for r in f.readlines()]
	final_res = []
	for i,v in enumerate(results):
		if (i+1) % 82 != 0:
			df.append([str((i+1) % 82) , v ,sets[set_idx]])
			# print(str((i+1) % 82) + "\t" + v + "\t" +sets[set_idx])
		else :
			final_res.append(v)
			set_idx+=1
	print("Final Results : ")
	print('\t'.join(final_res))
	df = pd.DataFrame(df, columns = ["iter", "map", "curriculum"])
	print(df)
	df.to_csv("eval_during_training_bert.csv", index=False)