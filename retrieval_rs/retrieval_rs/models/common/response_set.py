"""
    Class for SR response set refactored from existing code
    Stores the attributes of the response set and has some basic functions such as deduplication and mapping
    @author: Budhaditya Deb
"""
import os
from retrieval_rs.models.common.tokenization import Vocab, Tokenizer
from retrieval_rs.constants.file_names import RESPONSE_MAPPING_FILE_NAME, RESPONSE_SET_FILE_NAME, OUTPUT_RESPONSES_FILE_NAME

class ResponseSet:
    """
        This creates all attributes related to response set.
        There are essentially two types of members, Lists and Dicts.
        Lists are assumed to be accessed through response ID, while dicts are accessed through txt
        Args to set:
            --no_response_deduplication
            --no_response_mapping
            --rsp_input_dir
            --rsp_mapping_input_dir
        TODO: check if these are set correctly
    """
    def __init__(self, params):
        self.params = params
        self.tokenizer = Tokenizer(vocab_path=self.params.generic_vocab_input_dir, use_sentence_start_end=False)

        self.lm_scores_list = []
        self.rsp_cluster_ids_list = []
        self.raw_responses_list = []
        self.rsp_mapping_list = []
        self.tokenized_responses = []
        self.tokenized_mappings = []
        self.rsp_txt_to_rsp_mapping = {}

        self.read_attributes_from_rsp_files()
        self.tokenized_responses = [' '.join(self.tokenizer.tokenize(rsp)) for rsp in self.raw_responses_list]
        self.tokenized_mappings = [' '.join(self.tokenizer.tokenize(rsp)) for rsp in self.rsp_mapping_list]

    def read_attributes_from_rsp_files(self):
        """
            Inputs: Response Set and Response Mapping file directories
        """
        self.rsp_set_path = os.path.join(self.params.rsp_input_dir, RESPONSE_SET_FILE_NAME)
        with open(self.rsp_set_path, 'r', encoding='utf-8') as f_in:
            print("SystemLog: Reading response set LM/Clusters from %s"%self.rsp_set_path)
            for line in f_in:
                tokens = line.strip().split('\t')
                num_required_columns = max([self.params.rsp_lm_col, self.params.rsp_cluster_col, self.params.rsp_text_col])
                assert(len(tokens) >= num_required_columns), "SystemLog: ERROR reading response lm/cluster file. This file should have at least three tab separated columns for response, lm score and cluster ids"
                self.lm_scores_list.append(float(tokens[self.params.rsp_lm_col]))
                self.rsp_cluster_ids_list.append(tokens[self.params.rsp_cluster_col])
                self.raw_responses_list.append(tokens[self.params.rsp_text_col])

        if self.params.rsp_mapping_input_dir:
            self.rsp_mapping_path = os.path.join(self.params.rsp_mapping_input_dir, RESPONSE_MAPPING_FILE_NAME)
            with open(self.rsp_mapping_path, 'r', encoding='utf-8') as f_in:
                print("SystemLog: Reading response mapping from %s"%self.rsp_mapping_path)
                for line in f_in:
                    resp_mappings = line.strip().split('\t')
                    assert(len(resp_mappings) >= 2), "SystemLog: ERROR reading response mapping folder. This file should have two tab separated columns"
                    self.rsp_txt_to_rsp_mapping[resp_mappings[0]] = resp_mappings[1]
                    self.rsp_mapping_list.append(resp_mappings[1])
        else:
            # create dummy response mapping with the same text as the raw responses
            print("SystemLog: WARNING: Response mapping not provided. Creating default mapping from the raw responses")
            for item in self.raw_responses_list:
                self.rsp_txt_to_rsp_mapping[item] = item
                self.rsp_mapping_list.append(item)

        self.num_responses = len(self.raw_responses_list)
        assert (len(self.raw_responses_list) > 0), "SystemLog: Empty Response LM File"
        assert(self.num_responses == len(self.rsp_mapping_list)), "Number of responses in the Response Set file and the Mapping file do not match"
        assert(self.num_responses == len(set(self.raw_responses_list))), "Response Set might have duplicate responses. Please fix it"
        print("SystemLog: Added %d Responses in Response Set Class"%(self.num_responses))

    def get_clustered_responses(self, rsp_ids, rsp_scores, max_num_out_resps=3):
        """
            Deduplicates responses belonging to the same cluster
            Returns the top 3 responses after deduplication
            Args:
                rsp_ids: [batch_size, top_k]
                rsp_scores: [batch_size, top_k]
                max_num_out_resps: number of responses to return
            Returns:
                candidate_responses: text strings of responses [batch_size, max_num_out_resps]
                candidate_scores: of responses [batch_size, max_num_out_resps]
                candidate_ids: response indexes [batch_size, max_num_out_resps]
        """
        candidate_responses = []
        candidate_scores = []
        candidate_ids = []
        clusters = set()
        for response_score, response_idx in zip(rsp_scores, rsp_ids):
            if self.rsp_cluster_ids_list[response_idx] not in clusters:
                candidate_responses.append(self.raw_responses_list[response_idx])
                candidate_scores.append(response_score)
                candidate_ids.append(response_idx)
                clusters.add(self.rsp_cluster_ids_list[response_idx])
            if len(candidate_responses) == max_num_out_resps:
                break
        assert (len(set(candidate_responses)) == len(candidate_responses)), "SystemLog: Candidate responses are not all different"
        return candidate_responses, candidate_scores, candidate_ids

    def get_response_index(self, tokenized_rsp):
        """
            Args:
                tokenized_rsp: response string tokenized using generic tokenizer
            Returns:
                index of the response in the list
        """
        return self.tokenized_responses.index(tokenized_rsp)

    def get_response_text(self, response_ids):
        """
            Returns the corresponding response text for predicted IDs from model.
            Args:
                response_ids: a list of indexes
            Returns:
                responses: respective response text
        """
        responses = [self.raw_responses_list[ids] for ids in response_ids]
        return responses

    def get_tokenized_response(self, ind_or_text):
        """
            tokenized responses are used for computing the ROUGE and BLEU Scores
            Args:
                ind_or_text: either raw response or the index of the response
            Returns:
                tokenized_response: the tokenized string for the response

        """
        if type(ind_or_text) == str:
            # get the index of the Raw response string
            ind_or_text = self.raw_responses_list.index(ind_or_text)

        return self.tokenized_responses[ind_or_text]

    def get_tokenized_response_mapping(self, ind_or_text):
        """
            tokenized responses are used for computing the ROUGE and BLEU Scores
            Args:
                ind_or_text: either raw mapping text or the index of the response
            Returns:
                mapped responses: of the raw response text or index

        """
        if type(ind_or_text) == str:
            # get the index of the Raw response mapping
            ind_or_text = self.rsp_mapping_list.index(ind_or_text)
        return self.tokenized_mappings[ind_or_text]

    def deduplicate_and_map_responses(self, top_k_ids_list, top_k_vals_list, max_num_out_rsps=3):
        """
            Deduplicates responses belonging to the same cluster and maps responses using the editorial mapping
            Args:
                top_k_ids_list: topk ids of the predicted responses
                top_k_vals_list: topk ids of the predicted responses
                max_num_out_rsps: number of responses to return
            Returns:
                candidate_responses_list: Texxt of the predicted response
                candidate_scores_list: scores of the predicted response
                candidate_ids_list: IDs of the predicted responses
                candidate_responses_tokenized_list: response text tokenized using generic tokenizer
            TODO: do this through PyTorch graph later
        """
        top_k_ids_list = top_k_ids_list.tolist()
        top_k_vals_list = top_k_vals_list.tolist()
        candidate_responses_list = []
        candidate_scores_list = []
        candidate_ids_list = []
        candidate_responses_tokenized_list = []

        for index in range(len(top_k_ids_list)):
            candidate_ids = top_k_ids_list[index][0:max_num_out_rsps]
            candidate_scores = top_k_vals_list[index][0:max_num_out_rsps]
            candidate_responses = [self.raw_responses_list[id] for id in candidate_ids]
            candidate_responses_tokenized = [self.tokenized_responses[id] for id in candidate_ids]

            if self.params.response_deduplication == True:
                candidate_responses, candidate_scores, candidate_ids = self.get_clustered_responses(top_k_ids_list[index], top_k_vals_list[index], max_num_out_resps=3)
                candidate_responses_tokenized = [self.tokenized_responses[id] for id in candidate_ids]

            if self.params.response_mapping == True:
                candidate_responses = [self.rsp_mapping_list[id] for id in candidate_ids]
                candidate_responses_tokenized = [self.tokenized_mappings[id] for id in candidate_ids]

            candidate_ids_list.append(candidate_ids)
            candidate_scores_list.append(candidate_scores)
            candidate_responses_list.append(candidate_responses)
            candidate_responses_tokenized_list.append(candidate_responses_tokenized)
        return candidate_responses_list, candidate_scores_list, candidate_ids_list, candidate_responses_tokenized_list

    def export_response_file(self, rs_path, response_cluster_path, responses_path):
        """
            Converts response file as per Qas export specifications
        """
        print("SystemLog: Creating response cluster file at %s" % response_cluster_path)
        print("SystemLog: Creating responses file at %s" % responses_path)
        self.params.rsp_lm_col, self.params.rsp_cluster_col, self.params.rsp_text_col
        with open(rs_path, 'r', encoding='utf-8') as rs_in_file, \
                open(response_cluster_path, 'w', encoding='utf-8') as rs_cluster_out, \
                open(responses_path, 'w', encoding='utf-8') as rs_out:
            for line in rs_in_file:
                tokens = line.strip().split('\t')
                rs_cluster_out.write('{0}\t{1}\n'.format(tokens[self.params.rsp_text_col], tokens[self.params.rsp_cluster_col]))
                rs_out.write('{0}\n'.format(tokens[self.params.rsp_text_col]))
