import argparse
from IPython import embed
import pandas as pd
import os
from scipy import stats

path = "/tudelft.net/staff-umbrella/conversationalsearch/transformers_cl/pytorch_transformers_cl/data/"
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--dataset",
                    default="ms_v2",
                    type=str,
                    help="")

args = parser.parse_args()


path_dataset = path+args.dataset+"_output/"
final_str = ""
for seed in ['1','2','3','4','5']:
	baseline = pd.read_csv(path_dataset+'aps_run_cl__bert_avg_loss_c_3standard_training_seed_'+seed, names=['ap'])
	final_str += str(round(baseline.mean().values[0],4))+ "\t"
	for competing_pacing in ['geom_progression', 'step', 'linear', 'root_2', 'root_5', 'root_10', 'root_50']:
		competing = pd.read_csv(path_dataset+'aps_run_cl__bert_avg_loss_c_3'+competing_pacing+'_seed_'+seed, names=['ap'])		
		statistic, pvalue = stats.ttest_rel(competing['ap'].values, baseline['ap'].values)
		# statistic, pvalue = stats.wilcoxon(competing['ap'].values, baseline['ap'].values)
		p=""
		if pvalue<=0.01 and statistic >0 :
			p+=" $^{\\ddagger \\dagger}$"
		elif pvalue<=0.05 and statistic >0 :
			p+="$^{\\dagger}$"
		# if pvalue<=0.10 and statistic >0 :
			# p+="-"
		final_str+= (str(round(competing.mean().values[0],4))+" "+p+"\t")
	final_str+="\n"
		# print("\n")
		# embed()
print(final_str)