import math
import os
import csv
import numpy as np
import pandas as pd
from enum import Enum

from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import tensorflow as tf
import tensorflow_hub as hub

import torch
import sys
# change path after downloading InferSent code
sys.path.insert(0, 'path/to/InferSent')
from models import InferSent


class TextClassifierOptions(Enum):
    ENC_BASED_ON_BON = 1
    ENC_BASED_ON_INFERSENT = 2
    ENC_BASED_ON_TRANSFORMER = 3
    ENSAMBLE = 4
    ALL = 5
    

class Encoder1:
    ''' Bag of n-grams encoder'''
    
    VECTORIZER_MODEL = 'bon-vectorizer.joblib'
    FEATURE_SELECTON_MODEL = 'bon-selector-fpr.joblib'
    
    def __init__(self, models_dir):
        self.vectorizer = self.load_model(os.path.join(models_dir, self.VECTORIZER_MODEL))
        self.selector = self.load_model(os.path.join(models_dir, self.FEATURE_SELECTON_MODEL))
        
    def load_model(self, model_path):
        return joblib.load(model_path)
    
    def start(self, texts):
        pass
        
    def close(self):
        pass
        
    def encode(self, texts_batch):
        texts_batch_vec =  x = self.vectorizer.transform(texts_batch)
        texts_batch_vec = self.selector.transform(texts_batch_vec)
        return texts_batch_vec.astype(np.float64)
    
    
class Encoder2:   
    ''' Encoder based on InferSent '''
    
    WORD_VECTORS_FILE = 'crawl-300d-2M.vec'
    MODEL_FILE = 'infersent2.pkl'
    
    def __init__(self, word_vectors_dir, models_dir):
        word_vectors = os.path.join(word_vectors_dir, self.WORD_VECTORS_FILE)
        model_file =  os.path.join(models_dir, self.MODEL_FILE)

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(model_file))
        self.model.set_w2v_path(word_vectors)
      
    def start(self, texts):
        texts_list = texts.values.tolist()        
        self.model.build_vocab(texts_list, tokenize=True)
        
    def close(self):
        pass
    
    def encode(self, texts_batch):
        texts_batch_list = texts_batch.values.tolist()                    
        texts_batch_vec = self.model.encode(texts_batch_list, tokenize=True)
        
        return texts_batch_vec
    
    
class Encoder3:
    ''' Encoder based on Universal Sentence Encoder (Transformer) '''
    
    MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    
    def __init__(self, dan_model=False):
        self.embed = hub.Module(self.MODULE_URL)        
        self.messages = tf.placeholder(dtype=tf.string, shape=[None])
        self.output = self.embed(self.messages)
        
    def start(self, texts):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())
        
    def close(self):
        self.session.close()
    
    def encode(self, texts_batch):
        texts_batch_vec = self.session.run(self.output, feed_dict={self.messages: texts_batch.values.tolist()})
        return texts_batch_vec
    
    
class Classifier:
    
    def __init__(self, models_dir, classifier_name, model_path):
        self.classifier_name = classifier_name
        self.clf = self.load_model(os.path.join(models_dir, model_path))
        
    def load_model(self, model_path):
        return joblib.load(model_path)
    
    def predict(self, clf, X):
        predictions = clf.predict(X)
    
        if type(clf) is LinearSVC:
            predictions_proba = clf.decision_function(X)
        else:
            predictions_proba = clf.predict_proba(X)
    
        return pd.DataFrame(data={'y_predicted': predictions, 
                                  'probability/confidence(svm)': predictions_proba if type(clf) is LinearSVC else predictions_proba[:,1]})

    def run(self, input_data_vec):
        return self.predict(self.clf, input_data_vec)
    

class EnsambleClassifier:
    
    def __init__(self, classifier_name, threshold_percent):
        self.classifier_name = classifier_name
        self.threshold_percent = threshold_percent
    
    def predict(self, input_data_batch):
        y_predicted_columns = input_data_batch[input_data_batch.columns[input_data_batch.columns.to_series().str.contains('y_predicted')]]
        columns_count = len(y_predicted_columns.columns)        
        threshold = math.ceil(columns_count * self.threshold_percent / 100)
        
        y_sum = y_predicted_columns.sum(axis=1)
        y_predicted = np.where(y_sum < threshold, 0, 1)
        probability = y_sum / columns_count
           
        return pd.DataFrame(data={'y_predicted': y_predicted, 
                                  'probability/confidence(svm)': probability})

    def run(self, input_data_batch):
        return self.predict(input_data_batch)
    
    
class Pipeline:
    
    BATCH_SIZE = 5000
    
    def __init__(self, separator, pipeline_name, encoder, classifier):
        self.separator = separator
        self.pipeline_name = pipeline_name
        self.Encoder = encoder
        self.Classifier = classifier
    
    def write(self, data, full_data_file, append=True, header=True, encoding='utf-8'):
        if append:
            data.to_csv(full_data_file, sep=self.separator, encoding=encoding, index=False, mode='a', header=header)
        else:
            data.to_csv(full_data_file, sep=self.separator, encoding=encoding, index=False, header=header)
            
    def run(self, input_file):
        input_data = pd.read_csv(input_file, sep=self.separator, encoding='utf-8')
        
        if self.Encoder is not None:
            self.Encoder.start(input_data['text'])
        
        i = 0
        while i < input_data.shape[0]:
            start = i
            end = min(start+self.BATCH_SIZE, input_data.shape[0])
            print('%d - %d' % (start, end))
            input_data_batch = input_data.iloc[start:end]
            input_data_batch = input_data_batch.reset_index(drop=True)
            
            if self.Encoder is None:
                predicitons = self.Classifier.run(input_data_batch)
            else:
                input_data_batch_vec = self.Encoder.encode(input_data_batch['text'])
                predicitons = self.Classifier.run(input_data_batch_vec)
            
            input_data_batch['%s_y_predicted' % self.pipeline_name] = predicitons['y_predicted']
            input_data_batch['%s_probability/confidence(svm)' % self.pipeline_name] = predicitons['probability/confidence(svm)']

            self.write(input_data_batch, input_file, append=(start>0), header=(start==0))
            i = end
            
        if self.Encoder is not None:
            self.Encoder.close()
            
    
class TextClassifier:
    
    DECISION_TREE_NAME = 'decision_tree'
    LINEAR_SVM_1_NAME = 'linear_svm_1'
    LINEAR_SVM_2_NAME = 'linear_svm_2'
    ENSAMBLE_NAME = 'ensamble'
    
    DECISION_TREE_CLASSIFIER = 'decision-tree-classifier.joblib'
    LINEAR_SVM_CLASSIFIER_1 = 'linear-svm-classifier-1.joblib'
    LINEAR_SVM_CLASSIFIER_2 = 'linear-svm-classifier-2.joblib'
    
    FILTERED_FILE_SUFFIX = '-filtered.csv'
    
    def __init__(self, models_dir, word_vectors_dir, separator):
        self.separator = separator
        
        self.Encoder1 = Encoder1(models_dir)
        self.Encoder2 = Encoder2(word_vectors_dir, models_dir)
        self.Encoder3 = Encoder3()
        
        self.DecisionTreeClassifier = Classifier(models_dir, self.DECISION_TREE_NAME, self.DECISION_TREE_CLASSIFIER)
        self.DecisionTreePipeline = Pipeline(self.separator, self.DECISION_TREE_NAME, self.Encoder1, self.DecisionTreeClassifier)
        
        self.LinearSvmClassifier1 = Classifier(models_dir, self.LINEAR_SVM_1_NAME, self.LINEAR_SVM_CLASSIFIER_1)
        self.LinearSvmPipeline1 = Pipeline(self.separator, self.LINEAR_SVM_1_NAME, self.Encoder2, self.LinearSvmClassifier1)
        
        self.LinearSvmClassifier2 = Classifier(models_dir, self.LINEAR_SVM_2_NAME, self.LINEAR_SVM_CLASSIFIER_2)        
        self.LinearSvmPipeline2 = Pipeline(self.separator, self.LINEAR_SVM_2_NAME, self.Encoder3, self.LinearSvmClassifier2)

        self.EnsambleClassifier = EnsambleClassifier(self.ENSAMBLE_NAME, 50.0)
        self.EnsamblePipeline = Pipeline(self.separator, self.ENSAMBLE_NAME, None, self.EnsambleClassifier)
        
    def run(self, directory, option, merge=False, candidates_file=None):
        files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(self.FILTERED_FILE_SUFFIX)]
        for file in files:
            if option == TextClassifierOptions.ENC_BASED_ON_BON or option == TextClassifierOptions.ALL:
                self.DecisionTreePipeline.run(file)
            if option == TextClassifierOptions.ENC_BASED_ON_INFERSENT or option == TextClassifierOptions.ALL:
                self.LinearSvmPipeline1.run(file)
            if option == TextClassifierOptions.ENC_BASED_ON_TRANSFORMER or option == TextClassifierOptions.ALL:
                self.LinearSvmPipeline2.run(file)
            if option == TextClassifierOptions.ENSAMBLE or option == TextClassifierOptions.ALL:
                self.EnsamblePipeline.run(file)
                
        if merge:
            for i, file in enumerate(files):
                file_data = pd.read_csv(file, sep=self.separator, encoding='utf-8')
                file_data = file_data[file_data['ensamble_y_predicted'] == 1]
                file_data.to_csv(candidates_file, sep=self.separator, encoding='utf-8', index=False, mode='a', header=(i==0))
