# Paraphrasing
Seq2Seq supervised model 

This model is based on the following research paper:
https://www.aclweb.org/anthology/2020.coling-main.209/

## Use

This model could be used for supervised tasks where we have (input, target) training data: e.g. paraphrasing, summarisation, translation, QA 

## Dataset

We're using a subset of the following dataset:

@inproceedings{lan2017continuously,
  author     = {Lan, Wuwei and Qiu, Siyu and He, Hua and Xu, Wei},
  title      = {A Continuously Growing Dataset of Sentential Paraphrases},
  booktitle  = {Proceedings of The 2017 Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  year       = {2017},
  publisher  = {Association for Computational Linguistics},
  pages      = {1235--1245},
  location   = {Copenhagen, Denmark}
  url        = {http://aclweb.org/anthology/D17-1127}
} 

This dataset contains Twitter sentences and their paraphrased versions

## Training

Since project is for educational purposes, model was only loosely trained on 1 epoch of 1/5 of training data (30k sample), but we can see from the 'logs' folder that model is learning on training data, and also on evaluational data that's coming from different data distribution. Sufficient training for real-world applications would require more computational resources.

## Future updates
- Improving sampling algorithm & making it trainable
- Designing multi-objective training algorithm 
- Adding Flax model