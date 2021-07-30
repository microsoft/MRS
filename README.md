# Code for Multilingual Reply Suggestion (MRS)

This repository contains the code for our ACL 2021 paper:

Mozhi Zhang, Wei Wang, Budhaditya Deb, Guoqing Zheng, Milad Shokouhi, Ahmed Hassan Awadallah. [_A Dataset and Baselines for Multilingual Reply Suggestion_](https://arxiv.org/pdf/2106.02017.pdf).

If you find the repository useful, please cite:
```
@inproceedings{zhang-2021-mrs,
    title = {A Dataset and Baselines for Multilingual Reply Suggestion},
    author = {Mozhi Zhang and Wei Wang and Budhaditya Deb and Guoqing Zheng and Milad Shokouhi and Ahmed Hassan Awadallah},
    booktitle = {Proceedings of the Association for Computational Linguistics},
    doi = "10.18653/v1/2021.acl-long.97",
    year = {2021}
}
```

The code has three parts: an evaluation script (`eval.py`), retrieval model training (`retrieval_rs`), and generation model training (`generation_rs`). 

## Evaluate Models

Run the following to install dependencies:
```
pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'
```

The evaluation script `eval.py` takes a single argument:
```
python eval.py PRED_FILE
```
`PRED_FILE` is a TSV file. Each line is an example with the following columns:
```
Message <tab> Reference Reply <tab> Predicted Reply 1 <tab> Predicted Reply 2 <tab> Predicted Reply 3
```
For Japanese, add `--ja` to use the Japanese tokenizer.

## Train Retrieval Models

1. Install dependencies: `pip install -r retrieval_rs/requirements.txt`
2. (Optional) Install [Apex](https://github.com/NVIDIA/apex)
3. Download [multilingual BERT model](https://huggingface.co/bert-base-multilingual-cased) from huggingface
4. Use `retrieval_rs/train.sh` to train the model. You need to set the paths in the scripts.
5. Use `retrieval_rs/test.sh` to generate predictions and evaluate the model. You need to set the paths in the scripts.

## Train Generation Models

1. Install [Unicoder for generation](https://github.com/microsoft/Unicoder/tree/master/generation)
2. Download [Unicoder-xDAE model](https://1drv.ms/u/s!Amt8n9AJEyxckWbpMyGKPKWDjTG-?e=elsf31)
3. Use `generation_rs/preprocess.sh` to preprocess the model. You need to set the paths in the scripts.
4. Use `generation_rs/train.sh` to train the model. You need to set the paths in the scripts.
5. Use `generation_rs/test.sh` to generate predictions and evaluate the model. You need to set the paths in the scripts.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

