"""
    Utility functions similar to what is defined in common/utils.py
    There are slight modifications to each function because of different data readers which is why these are defined again.
    All these functions would eventually be merged with common/utils.py but requires considerable refactoring of the current MAtching model code and driver.
    TODO: Refactor these and merge with common/utils.py after refactoring the matching model code and driver.
    @author: Budhaditya Deb
"""
# coding=utf-8
import time
import os
import json
import sys
import pdb
import gc
import copy
import shutil
from itertools import islice
from operator import itemgetter


import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import importlib

from retrieval_rs.models.matching.data_processing import create_dataloaders, CollateMRSequence, get_dataset
from retrieval_rs.models.common.utils import compute_metrics, compute_rouge_n
from retrieval_rs.metric.rouge_n_ensemble import Sentence_ROUGE_n, Sentence_ROUGE_ensemble_f
from retrieval_rs.metric.rouge import Rouge
from retrieval_rs.constants.weights import SCORE_ERROR, UNIGRAM_WEIGHT, BIGRAM_WEIGHT, TRIGRAM_WEIGHT

IS_APEX_PRESENT = False
try:
    from apex.parallel import DistributedDataParallel as apex_DDP
    from apex import amp
    IS_APEX_PRESENT = True
except ImportError as e:
    print("SystemLog: Apex not found")

# TODO: Mode these to the another folder for distributed processing
def set_devices_ids_for_DDP(params):
    print('SystemLog: ---------------------------------------------------------')
    # use the AML option to get device IDs in AML. For running locally --node_rank sets it distributed.launch
    if params.aml:
        update_params_for_aml(params)

    print("SystemLog: Setting device as %s" % params.device)
    params.device = torch.device("cuda", params.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    print('SystemLog: Setting cuda to Local rank = {}'.format(params.local_rank))
    torch.cuda.set_device(params.local_rank)

def print_and_save_parameters(params):
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: ---------------------------------------------------------')
        print("SystemLog: Printing model parameters")
        filepath = os.path.join(params.model_output_dir, 'params.txt')
        with open(filepath, 'w', encoding='utf-8') as f_out:
            param_list = [(k, v) for k, v in vars(params).items()]
            param_list.sort(key=itemgetter(0))
            for k, v in param_list:
                print('SystemLog: %s\t%s' % (k, v))
                f_out.write('%s\t%s\n' % (k, v))

def compute_lm_sweep_window(current_lm_alpha, sweep_lm_span, window):
    sweep_range = []

    if current_lm_alpha <= window:
        if current_lm_alpha > 0:
            window = current_lm_alpha / 2.0
        else:
            window = window / 2.0
    low_lm_alpha = max(0, current_lm_alpha - sweep_lm_span * window)
    high_lm_alpha = current_lm_alpha + (sweep_lm_span + 1) * window # arange excluded the max value
    sweep_range = np.arange(low_lm_alpha, high_lm_alpha, window).tolist()
    return sweep_range

def compute_train_graph_metrics(params, trained_model, mr_dataloader_valid, mr_dataset_valid, metrics_logger, epoch, minibatch_number):
    """
        Compute loss and precision on a held out validation set using the train graph on a general MR set
    """
    total_items = 0
    valid_steps = 0.0
    total_items_to_validate = params.validation_batches * params.batch_size_validation
    max_data_size = len(mr_dataset_valid)
    if max_data_size < total_items_to_validate:
        total_items_to_validate = max_data_size
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: ---------------------------------------------------------')
        print('SystemLog: Computing Validation scores on %d items' % total_items_to_validate)

    for _, sample_batched in enumerate(mr_dataloader_valid):
        if sample_batched is not None:
            if len(sample_batched['rsp_batch_tuple'][0]) > 1:
                with torch.no_grad():
                    output = trained_model(sample_batched)  # need to update for matching_model and cvae
                    loss_dict = trained_model.loss(output)
                total_items += len(sample_batched['msg_batch_tuple'][0])  # corresponds to the message
                valid_steps += 1
                metrics_logger.update_validation_run_stats(len(sample_batched['rsp_batch_tuple'][0]), loss_dict, epoch, minibatch_number)
            if total_items >= total_items_to_validate:
                break
    sys.stdout.flush()
    metrics_logger.set_validation_running_stats(params, epoch, minibatch_number)
    return metrics_logger

def evaluate_metrics_on_gmr(params, train_model, gmr_dataloader, tokenizer):
    if params.local_rank == 0 and params.node_rank == 0:
        print("SystemLog: Evaluating MRR/P@K on GMR")
    # Compute MRR, P@K values on a golden MR set
    mrr = 0
    macro_mrr = 0
    mrr_f1 = 0
    avg_p_at_k = 0
    total = 0
    rsp_id_dict = {}  # for computing macro mrr
    # TODO: Let's not access the private members like this -- create a getter function instead
    for index in range(train_model.response_set.num_responses):
        rsp_id_dict[index] = [0.0, 0.0]  # set value and count to zero
    for i_batch, sample_batched in enumerate(gmr_dataloader):
        if  len(sample_batched['msg_batch_tuple'][0]) > 1:
            with torch.no_grad():
                top_k_vals, top_k_ids, _ = train_model(sample_batched)
                _, _, candidate_ids_list, _ = train_model.response_set.deduplicate_and_map_responses(top_k_ids, top_k_vals)
                top_k_ids = top_k_ids.tolist()
            if params.batch_size_infer == 1:
                # add batch axis if handling batch size of only 1 since the this is removed in forward
                top_k_ids = [top_k_ids]
            tokenized_responses = sample_batched['batch_replies'] # for gmr rsp is tokenized using generic_tokenizer inside the reader
            for index in range(len(tokenized_responses)):
                if tokenized_responses[index] in train_model.response_set.tokenized_responses:
                    if len(candidate_ids_list[index]) == 3:
                        golden_rsp_id = train_model.response_set.get_response_index(tokenized_responses[index])
                        rr, p_at_k = compute_metrics(golden_rsp_id, candidate_ids_list[index])
                        rsp_id_dict[golden_rsp_id] = [rsp_id_dict[golden_rsp_id][0] + rr, rsp_id_dict[golden_rsp_id][1] + 1]
                        mrr += rr
                        avg_p_at_k += p_at_k
                        total += 1
            if i_batch >= params.infer_batches:
                break
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: %d items in GMR metrics' % total)
    sys.stdout.flush()
    if total != 0:
        for v in rsp_id_dict.values():
            if v[1] > 0:
                macro_mrr += v[0] / v[1]
        mrr = mrr / total
        macro_mrr = macro_mrr / len(rsp_id_dict)
        mrr_f1 = 2 * mrr * macro_mrr / (mrr + macro_mrr)
        avg_p_at_k = avg_p_at_k / total

    gc.collect()
    gmr_metrics = {'mrr': mrr, 'macro_mrr': macro_mrr, 'mrr_f1': mrr_f1, 'p_at_k': avg_p_at_k}
    return gmr_metrics

def compute_rouge_n_for_responses(top_k_responses, k=3):
    rouge_n = []
    if len(top_k_responses) > 1:
        rouge_n.append(Rouge().get_scores(top_k_responses[0], top_k_responses[1])[0])
    if len(top_k_responses) > 2:
        rouge_n.append(Rouge().get_scores(top_k_responses[1], top_k_responses[2])[0])
        rouge_n.append(Rouge().get_scores(top_k_responses[2], top_k_responses[0])[0])
    return rouge_n

def evaluate_rouge_scores(params, train_model, mr_dataloader_valid=None, mr_dataset_valid=None, tokenizer=None):
    # TODO: Add Distributed Rouge computation as implemented in models/common/utils.py
    '''Compute all sentence-ROUGE metrics for all MR pairs on a validation set'''
    if params.local_rank == 0 and params.node_rank == 0:
        print("SystemLog: Evaluating ROUGE metrics on Validation Set")
    if mr_dataloader_valid is None:
        mr_dataset_valid, _ = get_dataset(params, run_mode="valid")
        mr_dataloader_valid = DataLoader(mr_dataset_valid, batch_size=params.batch_size_validation,
                                shuffle=False, num_workers=0, collate_fn=CollateMRSequence(params))
    # ---------- Initialize sentence-level ROUGE metrics ----------
    total = 0
    idx = 0
    all_metrics_relevance = {}
    metrics_relevance = []
    metrics_relevance_idx = []
    all_metrics_diversity = {}
    metrics_diversity = []
    metrics_diversity_idx = []

    for n in ["1", "2", "3", "L"]:
        metrics_relevance_idx.append(idx)
        metrics_diversity_idx.append(idx)
        for metric in ["f", "p", "r"]:
            metrics_relevance.append(Sentence_ROUGE_n(n, metric))
            metrics_diversity.append(Sentence_ROUGE_n(n, metric))
            idx += 1
    # ---------- Initialize ensembled sentence-level ROUGE metrics ----------
    ensemble_metrics_relevance = [Sentence_ROUGE_ensemble_f("uniform"), Sentence_ROUGE_ensemble_f("weight")]
    ensemble_metrics_diversity = [Sentence_ROUGE_ensemble_f("uniform"), Sentence_ROUGE_ensemble_f("weight")]
    for i_batch, sample_batched in enumerate(mr_dataloader_valid):
        if sample_batched is not None and len(sample_batched['rsp_batch_tuple'][0]) > 1:
            top_k_vals, top_k_ids, _ = train_model(sample_batched)
            _, _, _, candidate_responses_tokenized_list = train_model.response_set.deduplicate_and_map_responses(top_k_ids, top_k_vals)
            top_k_ids = top_k_ids.tolist()
            if params.batch_size_infer == 1:
                # add batch axis if handling batch size of only 1 since the this is removed in forward
                top_k_ids = [top_k_ids]
            tokenized_responses = sample_batched['batch_replies'] # responses are tokenized inside the reader using generic for valid dataset
            for index in range(len(tokenized_responses)):
                if len(candidate_responses_tokenized_list[index]) == 3:
                    # ----------- truncate reference responses -------------
                    response_tokens = tokenized_responses[index]
                    # if params.max_golden_resp_length_to_trim_start_for_rouge > -1:
                    #     response_tokens = response_tokens[:params.max_golden_resp_length_to_trim_start_for_rouge]
                    rouge_n = compute_rouge_n(response_tokens, candidate_responses_tokenized_list[index])
                    rouge_n_responses = compute_rouge_n_for_responses(candidate_responses_tokenized_list[index])
                    rouge_n_scores = []
                    rouge_n_scores_responses = []
                    for metric in metrics_relevance:
                        rouge_n_scores.append(metric.compute_add(rouge_n))
                    for metric in metrics_diversity:
                        rouge_n_scores_responses.append(metric.compute_add(rouge_n_responses))
                    for ens_metric in ensemble_metrics_relevance:
                        ens_metric.compute_add(rouge_n_scores, metrics_relevance_idx[0], metrics_relevance_idx[1], metrics_relevance_idx[2], UNIGRAM_WEIGHT, BIGRAM_WEIGHT, TRIGRAM_WEIGHT)
                        all_metrics_relevance[ens_metric.name] = ens_metric.aggregate()
                    for ens_metric in ensemble_metrics_diversity:
                        ens_metric.compute_add(rouge_n_scores_responses, metrics_diversity_idx[0], metrics_diversity_idx[1], metrics_diversity_idx[2], UNIGRAM_WEIGHT, BIGRAM_WEIGHT, TRIGRAM_WEIGHT)
                        all_metrics_diversity[ens_metric.name] = ens_metric.aggregate()
                    total += 1
            if i_batch >= params.validation_batches:
                break
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: %d items in ROUGE metrics' % total)
        sys.stdout.flush()
    gc.collect()
    all_metrics = all_metrics_relevance
    all_metrics['Diversity_ROUGE_uniform_f'] = all_metrics_diversity['Sentence_ROUGE_uniform_f']
    all_metrics['Diversity_ROUGE_weight_f'] = all_metrics_diversity['Sentence_ROUGE_weight_f']
    return all_metrics

def sweep_lm_alpha_for_metrics(params, train_model, metrics_logger, epoch, minibatch, gmr_dataloader, mr_dataloader_valid, tokenizer):
    # Sweep over different lm alpha values to find the best model metrics
    # Use the current lm_alpha value and sweep over a small window (+-window) to get the best metrics
    # In the next evaluation the current lm_alpha value is switched to the latest best and search is over a new window if updated
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: ---------------------------------------------------------')
    #  restrict LM sweep to over just the Rouge Metrics.
    # all_metrics = ['Sentence_ROUGE_uniform_f', 'Sentence_ROUGE_weight_f']
    all_metrics = ['Sentence_ROUGE_weight_f']
    sweep_steps = []
    if params.sweep_lm is False:
        sweep_steps = [params.lm_alpha]
    else:
        for metric in all_metrics:
            sweep_steps.extend(compute_lm_sweep_window(metrics_logger.valid_metrics[metric].best_lm_alpha, params.sweep_lm_span, params.sweep_lm_window))

    sweep_steps = [round(a, 2) for a in sweep_steps]
    sweep_steps = sorted(list(set(sweep_steps)))

    best_mrr = 0
    best_macro_mrr = 0
    best_p_at_k = 0
    best_mrr_f1 = 0
    best_lm_alpha = params.lm_alpha
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: Sweeping for lm_alpha values between %s' % sweep_steps)
    for lm_alpha in sweep_steps:
        train_model.set_lm_alpha(lm_alpha)
        train_model.eval()
        # ---------- metrics on GMR set -----------
        gmr_metrics = evaluate_metrics_on_gmr(params, train_model, gmr_dataloader, tokenizer)
        # ---------- metrics on Validation set with general MR pairs -----------
        rouge_metrics = evaluate_rouge_scores(params, train_model, mr_dataloader_valid, tokenizer=tokenizer)

        if params.local_rank == 0 and params.node_rank == 0:
            print("SystemLog: -----------------------------------------------------------------------------------")
            print("SystemLog:\tlm_alpha\tMRR\tMacroMRR\tMRRF1\tP@k\tSentence_ROUGE_uniform_f\tSentence_ROUGE_weight_f\tDiversity_ROUGE_uniform\tDiversity_ROUGE_weight")
            print("SystemLog:\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"%(
                    lm_alpha,
                    gmr_metrics['mrr'],
                    gmr_metrics['macro_mrr'],
                    gmr_metrics['mrr_f1'],
                    gmr_metrics['p_at_k'],
                    rouge_metrics['Sentence_ROUGE_uniform_f'],
                    rouge_metrics['Sentence_ROUGE_weight_f'],
                    rouge_metrics['Diversity_ROUGE_uniform_f'],
                    rouge_metrics['Diversity_ROUGE_weight_f'])
                    )
            print("SystemLog: -----------------------------------------------------------------------------------")
            sys.stdout.flush()

        if gmr_metrics['mrr_f1'] > best_mrr_f1:
            best_lm_alpha = lm_alpha
        best_mrr = max(gmr_metrics['mrr'], best_mrr)
        best_mrr_f1 = max(gmr_metrics['mrr_f1'], best_mrr_f1)
        best_p_at_k = max(gmr_metrics['p_at_k'], best_p_at_k)
        best_macro_mrr = max(gmr_metrics['macro_mrr'], best_macro_mrr)

        metrics_logger.update_validation_metrics(epoch, minibatch, lm_alpha, gmr_metrics)
        metrics_logger.update_validation_metrics(epoch, minibatch, lm_alpha, rouge_metrics)
        # TODO: Dont add this for now as the names of the metrics are the same for relevance and diversity. Need to change it later.
        # metrics_logger.update_validation_metrics(epoch, minibatch, lm_alpha, rouge_metrics_diversity)

    return best_mrr, best_macro_mrr, best_mrr_f1, best_p_at_k, best_lm_alpha

def predict_responses_from_list(params, train_model, mr_dataset_valid):
    """ This function is used to print responses for specific messages during training time
        Arguments:
        params: Parameters of the model
        inference_model: Inference graph which we want to run
        mr_dataset_valid: Used to get message token tuples for running inference graph.
        Returns: None
    """
    message_list = [
        'Good morning!',
        'That was really funny!',
        'Hello, how are you?',
        'Can you meet me for lunch?',
        'I am going for a vacation!',
        'Can you send me the document?',
        'Here are the latest updates.',
        'I am not feeling very well.',
        'You have a 6:30 AM appointment tomorrow morning.',
        'I am on my way',
        'Finished writing the report and fixed the issue.',
        'What do you want to eat?',
        'It is such a nice sunny day!',
        'Lets go to the beach',
        'We need a bigger boat',
        'Tommorow is another day',
        'Frankly my dear, I dont give a damn!',
        'I will be back!',
        'This is sparta!'
        ]

    collate_funct = CollateMRSequence(params)
    msg_batch = []
    for msg in message_list:
        line = [""] * (max(params.msg_col, params.rsp_col, params.rsp_label_col)+1)
        line[params.msg_col] = msg
        line[params.rsp_col] = '_NA_'
        line = '\t'.join(line)
        msg_batch.append(mr_dataset_valid.datasets[0].get_line_items(line))

    batched_messages = collate_funct(msg_batch)
    top_k_vals_list, top_k_ids_list, _ = train_model(batched_messages)
    candidate_responses_list, _, _, _ = train_model.response_set.deduplicate_and_map_responses(top_k_ids_list, top_k_vals_list)
    for index in range(len(candidate_responses_list)):
        top_3_responses = '   '.join(candidate_responses_list[index])
        if params.local_rank == 0 and params.node_rank == 0:
            print("SystemLog: ", message_list[index], '\t', top_3_responses.encode('utf-8'))
    if params.local_rank == 0 and params.node_rank == 0:
        print('SystemLog: ---------------------------------------------------------')

def validation_loop(params, train_model, mr_dataloader_valid, mr_dataset_valid, gmr_dataloader, tokenizer, metrics_logger, epoch, i_batch):
    with torch.no_grad():
        train_model.eval()
        train_model.set_run_mode(run_mode="valid")
        # TODO: currently not working, will fix it later but it is not used for tracking any metrics right now
        # metrics_logger = compute_train_graph_metrics(params, train_model, mr_dataloader_valid, mr_dataset_valid, metrics_logger, epoch, i_batch+1)
        # -------------- predict for some example messages -------------
        train_model.set_run_mode(run_mode="eval")
        predict_responses_from_list(params, train_model, mr_dataset_valid)
        # # --------Compute BLEU on the inference graph --------
        # evaluate_bleu_scores(params, train_model, mr_dataloader_valid, mr_dataset_valid, metrics_logger, epoch, i_batch + 1)
        # -------- Compute mrr and p@k on the inference graph --------
        mrr, macro_mrr, mrr_f1, p_at_k, lm_alpha = sweep_lm_alpha_for_metrics(params, train_model, metrics_logger, epoch, i_batch+1, gmr_dataloader, mr_dataloader_valid, tokenizer)
        # ----------------- best metrics -----------
        if params.local_rank == 0 and params.node_rank == 0:
            print('SystemLog: ---------------------------------------------------------')
            metrics_logger.print_validation_metrics()
            print('SystemLog: ---------------------------------------------------------')
        metrics_dict = metrics_logger.get_best_valid_metric()
        best_loss = metrics_dict['loss']
        best_precision = metrics_dict['precision']
        best_mrr = metrics_dict['mrr_f1']
        # ------- reset the train model to train -----------
        train_model.set_run_mode(run_mode="train")
        train_model.train()
    gc.collect()
    return metrics_logger, best_mrr, best_loss, best_precision

