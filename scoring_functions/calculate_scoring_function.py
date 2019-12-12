import argparse
import numpy as np
import pandas as pd
from IPython import embed
from conv_curriculum_helper import load_dataset, save_curriculum, ap
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from gensim.summarization.bm25 import BM25

SEMANTIC_SIM_N_WORDS = 20
import random

def split_into_buckets(values, n_buckets, invert_order=False):

    c = [v for v in zip(values, range(len(values)))]
    c = sorted(c, key=lambda x:x[0], reverse=invert_order)

    assert n_buckets == 3

    idx_cut_1 = int(0.33 * len(values))
    idx_cut_2 = int(0.66 * len(values))

    idx_to_bucket = {}
    for v, idx in c[0:idx_cut_1]:
        idx_to_bucket[idx] = 0
    print('33 percentile: '+ str(v))
    for v, idx in c[idx_cut_1:idx_cut_2]:
        idx_to_bucket[idx] = 1    
    print('66 percentile: '+ str(v))
    for v, idx in c[idx_cut_2:]:
        idx_to_bucket[idx] = 2
    
    splited = []
    for i in range(len(values)):
        splited.append(idx_to_bucket[i])
    assert(len(values) == len(splited))
    print("Distribution: ")
    print(pd.DataFrame(splited).groupby(0)[0].count())
    print(len(splited))
    print("====== Finished\n")
    return splited



# RANDOM curriculum
def random_curriculum(dataset):
    random_by_q = {}
    
    for _, q, _ in dataset:
        query = ' '.join(q)
        if query not in random_by_q:
            random_by_q[query] = random.randint(1,101)

    return [random_by_q[' '.join(q)] for _, q, _ in dataset]

#============================================#
#                    QUERY                   #
#============================================#

def q_num_turns_curriculum(dataset):
    """ calculates the curriculum based on the number
    of turns the conversation has """
    print("====== Calculating number of turns curriculum")
    turns = []

    for _, q, _ in dataset:
        turns.append(len(q))

    return turns

def q_avg_num_words_curriculum(dataset):
    """ calculates the curriculum based the average 
    number of words in utterances"""
    print("====== Calculating avg num words in utterances curriculum")
    avg_words = []

    for _, queries, _ in dataset:        
        avg_words.append(np.mean([len(q.split(" ")) for q in queries]))

    return avg_words

#============================================#
#                    DOC                     #
#============================================#

def d_avg_num_words_curriculum(dataset):
    """ curriculum based on avg number of words in candidate docs""" 
    print("====== Calculating avg num words in candidate docs")
    values = []

    q_docs = {}
    for _, q , doc in dataset:
        query = ' '.join(q)
        if(query not in q_docs):
            q_docs[query] = []
        q_docs[query].append(len(doc.split(" ")))

    for _, q , doc in dataset:
        query = ' '.join(q)
        values.append(np.mean(q_docs[query]))

    return values

# #============================================#
# #              QUERY and DOC                 #
# #============================================#

def semantic_match_curriculum(dataset):
    """ std SM (Semantic Match) between q and docs """
    print("====== Calculating semantic match curriculum")
    c = []

    q_docs = {}
    for _, q , doc in dataset:
        query = ' '.join(q)
        if(query not in q_docs):
            q_docs[query] = []
        q_docs[query].append(doc)

    # fasttext_wikipedia
    wvecs = KeyedVectors.load_word2vec_format("./data/cc.en.300.vec",
        binary=False) 

    std_semantic_match = {}
    for q in tqdm([k for k in q_docs.keys()]):
        similarities = []
        for doc in q_docs[q]:
            sim = []
            for word_q in q.split(" ")[-SEMANTIC_SIM_N_WORDS:]:
                for word_doc in doc.split(" ")[0:SEMANTIC_SIM_N_WORDS]:
                    if word_doc in wvecs.vocab and word_q in wvecs.vocab:
                        sim.append(cosine_similarity([wvecs[word_q]],[wvecs[word_doc]])[0][0])
            if(len(sim) == 0):
                similarities.append(0)
            else:
                similarities.append(np.mean(sim))
       
        std_sim = np.std(similarities)
        std_semantic_match[q] = std_sim

    for _, q , doc in dataset:
        query = ' '.join(q)
        c.append(std_semantic_match[query])

    return c

def exact_match_curriculum(dataset):
    """ std BM25 between q and docs """
    print("====== Calculating BM25 match curriculum")
    c = []
    
    corpus = [doc.split(" ") for _, _ , doc in dataset]
    bm25 = BM25(corpus)

    q_bm25_scores = {}
    for i, (_, q , doc) in enumerate(tqdm(dataset)):
        query = ' '.join(q)
        if query not in q_bm25_scores:
            q_bm25_scores[query] = []    
        q_bm25_scores[query].append(bm25.get_score(query.split(" "), i))

    bm25_std = {}
    for query in q_bm25_scores.keys():
        bm25_std[query] = np.std(q_bm25_scores[query])

    for _, q , doc in dataset:
        query = ' '.join(q)
        c.append(bm25_std[query])
    return c

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv train file in format \" label \t utt1 \t utt2 ... \t uttN \t response \"")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The output_path containing curriculum labels for each instance in the input_file")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)

    number_of_buckets = 3

    turns_c = q_num_turns_curriculum(dataset)
    save_curriculum(turns_c, args.output_path + "_turns_c_values_" + str(number_of_buckets))
    labeled_turns_c = split_into_buckets(turns_c, number_of_buckets)
    # save_curriculum(labeled_turns_c, args.output_path + "_turns_c_" + str(number_of_buckets))

    avg_words_c = q_avg_num_words_curriculum(dataset)
    save_curriculum(avg_words_c, args.output_path + "_utt_avg_words_c_values_"+str(number_of_buckets))
    labeled_avg_words_c = split_into_buckets(avg_words_c, number_of_buckets)
    # save_curriculum(labeled_avg_words_c, args.output_path + "_utt_avg_words_c_"+str(number_of_buckets))

    cand_docs_avg_words_c = d_avg_num_words_curriculum(dataset)
    save_curriculum(cand_docs_avg_words_c, args.output_path + "_cand_docs_avg_words_c_values_"+str(number_of_buckets))
    labeled_cand_docs_avg_words_c = split_into_buckets(cand_docs_avg_words_c, number_of_buckets)
    # save_curriculum(labeled_cand_docs_avg_words_c, args.output_path + "_cand_docs_avg_words_c_"+str(number_of_buckets))

    avg_exact_match_q_d_c = exact_match_curriculum(dataset)
    save_curriculum(avg_exact_match_q_d_c, args.output_path + "max_dif_exact_match_q_d_c_values_"+str(number_of_buckets))
    labeled_avg_exact_match_q_d_c = split_into_buckets(avg_exact_match_q_d_c, number_of_buckets)
    # save_curriculum(labeled_avg_exact_match_q_d_c, args.output_path + "max_dif_exact_match_q_d_c_"+str(number_of_buckets))

    avg_semantic_match_q_d_c = semantic_match_curriculum(dataset)
    save_curriculum(avg_semantic_match_q_d_c, args.output_path + "max_dif_semantic_match_q_d_c_values_"+str(number_of_buckets))
    labeled_avg_semantic_match_q_d_c = split_into_buckets(avg_semantic_match_q_d_c, number_of_buckets)
    # save_curriculum(labeled_avg_semantic_match_q_d_c, args.output_path + "max_dif_semantic_match_q_d_c_"+str(number_of_buckets))

    # task = args.output_path.split("data/")[1]
    # ns_n = 2

    # with open("./data/"+task+"_bert_avg_pred_scores_c_values_3", 'r') as f:
    #     bert_avg_scores = [float(l.strip()) for l in f.readlines()]
    #     bert_avg_scores_for_each_doc = []
    #     for score in bert_avg_scores:
    #         for i in range(0, ns_n): # this changes depending on the number of negative sampled documents
    #             bert_avg_scores_for_each_doc.append(score)
    #     bert_avg_c = split_into_buckets(bert_avg_scores_for_each_doc, number_of_buckets, True)
    #     save_curriculum(bert_avg_c, args.output_path + "_bert_avg_pred_scores_c_"+str(number_of_buckets))

    # with open("./data/"+task+"_bert_avg_loss_c_values_3", 'r') as f:
    #     bert_avg_loss = [float(l.strip()) for l in f.readlines()]
    #     bert_avg_loss_for_each_doc = []
    #     for score in bert_avg_loss:
    #         for i in range(0, ns_n): # this changes depending on the number of negative sampled documents
    #             bert_avg_loss_for_each_doc.append(score)
    #     bert_avg_loss_c = split_into_buckets(bert_avg_loss_for_each_doc, number_of_buckets)
    #     save_curriculum(bert_avg_loss_c, args.output_path + "_bert_avg_loss_c_"+str(number_of_buckets))


if __name__ == "__main__":
    main()