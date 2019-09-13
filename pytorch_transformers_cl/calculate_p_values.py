import argparse
from IPython import embed
import pandas as pd
import os
from scipy import stats

path = "/tudelft.net/staff-umbrella/domaincss/wsdm19/bert_ranker/data/"
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--dataset",
                    default="ms_v2",
                    type=str,
                    help="")

args = parser.parse_args()

path_dataset = path+args.dataset+"_output/"
final_str = ""
pacing_func = "root_2"
for seed in ['1','2','3','4','5']:
	baseline = pd.read_csv(path_dataset+'aps_run_cl__random_c_3'+pacing_func+'_seed_'+seed, names=['ap'])
	final_str += str(round(baseline.mean().values[0],4))+ "\t"
	for competing_scoring in ['_turns_c_3'+pacing_func, '_utt_avg_words_c_3'+pacing_func,\
						'_cand_docs_avg_words_c_3'+pacing_func, 'max_dif_exact_match_q_d_c_3'+pacing_func,\
						'max_dif_semantic_match_q_d_c_3'+pacing_func, '_bert_preds_dif_c_3'+pacing_func,\
						'_bert_avg_loss_c_3'+pacing_func]:
		competing = pd.read_csv(path_dataset+'aps_run_cl_'+competing_scoring+'_seed_'+seed, names=['ap'])		
		statistic, pvalue = stats.ttest_rel(competing['ap'].values, baseline['ap'].values)
		# statistic, pvalue = stats.wilcoxon(competing['ap'].values, baseline['ap'].values)
		p=""
		if pvalue<=0.05 and statistic >0 :
			p+="$^{\\dagger}$"
		if pvalue<=0.01 and statistic >0 :
			p+="$^{\\ddagger}$"
		# if pvalue<=0.10 and statistic >0 :
			# p+="-"
		final_str+= str(round(competing.mean().values[0],4))
		final_str+= " "+p+"\t"


	final_str+="\n"
		# print("\n")
		# embed()
print(final_str)