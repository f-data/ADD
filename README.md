# ADD: Academic Disciplines Detector Based on Wikipedia
This repository contains the code and evaluation results of the research paper “ADD: Academic Disciplines Detector Based on Wikipedia”. The purpose of the Academic Disciplines Detector (ADD) is detection of academic disciplines defined in Wikipedia at particular moment, in order to facilitate the timely detection of emerging or obsolete disciplines and enable studying of their evolution.

## Citing
A. Gjorgjevikj, K. Mishev and D. Trajanov, "ADD: Academic Disciplines Detector Based on Wikipedia," in *IEEE Access*, vol. 8, pp. 7005-7019, 2020.
```
@article{ADD,
  author={A. {Gjorgjevikj} and K. {Mishev} and D. {Trajanov}},
  journal={IEEE Access},
  title={ADD: Academic Disciplines Detector Based on Wikipedia},
  year={2020},
  volume={8},
  pages={7005-7019},
  doi={10.1109/ACCESS.2019.2963674},
  ISSN={2169-3536},
}
```

## Requirements

- Python 3
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Gensim](https://github.com/RaRe-Technologies/gensim)
- [Scikit-learn](https://scikit-learn.org)
- [NetworkX](https://networkx.github.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Tensorflow Hub](https://www.tensorflow.org/hub)
- [Pytorch](https://pytorch.org/)
- [InferSent](https://github.com/facebookresearch/InferSent) [[1]](#1)


## Data and Models
The code available in this repository uses the following pretrained word embeddings and models:
1. [FastText](https://fasttext.cc/docs/en/english-vectors.html) [[2]](#2) word embeddings trained on Common Crawl (2 million word vectors).
2. [InferSent](https://github.com/facebookresearch/InferSent) model trained with fastText word embeddings (version 2).
3. [Universal Sentence Encoder (USE) - Transformer](https://tfhub.dev/google/universal-sentence-encoder-large/3) [[3]](#3) available in TensorFlow Hub.

The Academic Disciplines Detector (ADD) custom models were trained/evaluated with datasets based on:
1. Wikipedia export containing current revisions of all English articles (XML files *enwiki-YYYYMMDD-pages-articles.xml*), from 2017-06-01 for training, and exports from 2015-06-02, 2016-06-01, 2017-06-01, 2018-11-20 for evaluation.
2. Classification of Instructional Programs (CIP) data from 2010 (title and description) for training.


## Modules
To run each of the Academic Disciplines Detector (ADD) modules, see [modules/demo.ipynb](modules/demo.ipynb). The modules should be run in the specified order.

### Text/Metadata Extractor
The Extractor class reads Wikipedia XML export files and produces JSON files containing the extracted metadata and text. The preprocess argument controls whether to apply an optional preprocessing before the metadata and text extraction. The preprocessing code is provided for completeness only, as it was applied to get the results reported in the paper, but may not be needed or may need modifications depending on the Wikipedia exports and Gensim version.

***Usage***
1. Copy Wikipedia dump files in the data folder.
2. Run the code from [modules/demo.ipynb](modules/demo.ipynb).


### Basic Filter / Lead Section Excerpts Extractor
Filters Wikipedia articles that do not follow the patterns common for academic disciplines titles and extracts short representative subsection from the article’s lead section.

***Usage***
1. Make sure that Text/Metadata Extractor’s output files are present in the data folder.
2. Run the code from [modules/demo.ipynb](modules/demo.ipynb).


### Text Classifier
Calculates the probability that one Wikipedia article is an academic discipline based on trained classifiers over Wikipedia articles’ lead section excerpt.

***Usage***
1. Download the InferSent source code from its GitHub repository and specify the path to its models.py script in [modules/text_classifier.py](modules/text_classifier.py). InferSent is distributed by Facebook under the Creative Commons Attribution-NonCommercial 4.0 International Public License (for more information please see the [InferSent GitHub repository](https://github.com/facebookresearch/InferSent)).
2. Download the pretrained InferSent model (version 2) and copy it in the directory specified by the parameter MODELS_DIR.
3. Download the pretrained fastText word embeddings trained on Common Crawl (2 million word vectors) and copy it in the directory specified by the parameter WORD_VECTORS_DIR.
4. Make sure that Basic Filter’s output files are present in the data folder.
5. Run the code from [modules/demo.ipynb](modules/demo.ipynb). If the option TextClassifierOptions.ALL is not used, to reproduce the results, make sure to run the other three classifiers before running with the option TextClassifierOptions.ENSAMBLE.

### Node Classifier
Calculates the probability that one candidate discipline is an academic discipline based on a trained classifier over disciplines graph centrality-based features.

***Usage***
1. Make sure that Text Classifier’s merged output CSV file present in the data folder.
2. Run the code from [modules/demo.ipynb](modules/demo.ipynb). The final CSV file contains a probability score that a candidate Wikipedia article is an academic discipline.


## Evaluation
The test dataset errors made by the trained text classification and node classification models are available in the [evaluation](evaluation/) directory.

1. Text classifier: [ensamble-classifier-test-errors.csv](evaluation/ensamble-classifier-test-errors.csv)
2. Node classifier: [node-classifier-test-errors.csv](evaluation/node-classifier-test-errors.csv)

The 100 detected academic disciplines with highest probability score by Wikipedia export are available in the same directory. These are the labelled disciplines in the paper visualizations.

1. Export from 2015-06-02: [enwiki-20150602-detected-disciplines-limit-100.csv](evaluation/enwiki-20150602-detected-disciplines-limit-100.csv)
2. Export from 2016-06-01: [enwiki-20160601-detected-disciplines-limit-100.csv](evaluation/enwiki-20160601-detected-disciplines-limit-100.csv)
3. Export from 2017-06-01: [enwiki-20170601-detected-disciplines-limit-100.csv](evaluation/enwiki-20170601-detected-disciplines-limit-100.csv)
4. Export from 2018-11-20: [enwiki-20181120-detected-disciplines-limit-100.csv](evaluation/enwiki-20181120-detected-disciplines-limit-100.csv)


## References
<a id="1">[1]</a> A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes, "Supervised learning of universal sentence representations from natural language inference data," 2017, *arXiv:1705.02364*. [Online]. Available: https://arxiv.org/abs/1705.02364

<a id="2">[2]</a> T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, and A. Joulin, "Advances in pre-training distributed word representations," in *Proc. Int. Conf. Lang. Resour. Eval. (LREC)*, 2018.

<a id="3">[3]</a> D. Cer, Y. Yang, S.-Y. Kong, N. Hua, N. Limtiaco, R. S. John, N. Constant, M. Guajardo-Cespedes, S.Yuan, and C. Tar, "Universal sentence encoder," 2018, *arXiv:1803.11175*. [Online]. Available: https://arxiv.org/abs/1803.11175


## Note
The textual content coming from Wikipedia dumps is available under the GNU Free Documentation License (GFDL) and the Creative Commons Attribution-Share-Alike 3.0 License. For more information see the [License information about Wikimedia dump downloads](https://dumps.wikimedia.org/legal.html).
