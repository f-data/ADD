import os
import json
import csv
import pandas as pd
import networkx as nx
from smart_open import smart_open
from sklearn.externals import joblib


class GraphCreator:
    '''
    The GraphCreator creates csv file with links between the candidate Wikipedia articles.
    
    Arguments:
        separator: Csv separator
    '''
    def __init__(self, separator):
        self.separator = separator
                
    def append(self, data, file, append=True):
        writer = csv.writer(open(file, 'a', newline='', encoding='utf-8'), delimiter=self.separator)
        for row in data:
            writer.writerow(row)
            
    def save_edges(self, json_dir, candidates_file, graph_file):
        '''
        Extracting graph edges, i.e., links between a Wikipedia article (source) and all other articles it references (targets) in the text.
    
        Arguments:
            json_dir: string, path to the directory containing the json files
            candidates_file: string, path to the csv file with the candidates data
            graph_file: string, output file
        '''
        edges = []
        edges.append(['source', 'target'])
        
        candidates_data = pd.read_csv(candidates_file, sep=self.separator, encoding='utf-8')
        nodes = list(candidates_data['title'])
        nodes_l = [x.lower() for x in list(candidates_data['title'])]
        
        file_names = [file for file in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, file)) and file.endswith(".gz")]
        for file_name in file_names:
            print(file_name)
            for line in smart_open(os.path.join(json_dir, file_name)):
                article = json.loads(line)
                source = article['title'].lower()
                if source in nodes_l:
                    source_index = nodes_l.index(source)
                    source_title = nodes[source_index]
                    interlinks = article['interlinks']
                    for target in interlinks:
                        target = target.lower()
                        if target in nodes_l:
                            target_index = nodes_l.index(target)
                            target_title = nodes[target_index]
                            edges.append([source_title, target_title])
            self.append(edges, graph_file)
            edges = []
            
            
class FeatureExtractor:
    '''
    The FeatureExtractor calculates the features of each node of the graph created based on the provided edges list.
    
    Arguments:
        separator: CSV separator
    '''
    
    TEXT_COLUMNS = ['title', 'decision_tree_probability/confidence(svm)', 'linear_svm_1_probability/confidence(svm)', 'linear_svm_2_probability/confidence(svm)']
    
    def __init__(self, separator):
        self.separator = separator
        
    def calculate_gen_centralities(self, G, U, suffix):
        ''' Calculate centralities and add them as node attributes '''
         
        nx.set_node_attributes(G, nx.closeness_centrality(U, u=None, distance=None, wf_improved=True),  'closeness_%s' % suffix)
        nx.set_node_attributes(G, nx.betweenness_centrality(U, k=None, normalized=True, weight=None, endpoints=False, seed=None), 'betweenness_%s' % suffix)
        nx.set_node_attributes(G, nx.load_centrality(U, v=None, cutoff=None, normalized=True, weight=None), 'load_%s' % suffix)
        nx.set_node_attributes(G, nx.harmonic_centrality(U, nbunch=None, distance=None), 'harmonic_%s' % suffix)      
        return G
    
    def calculate_dir_centralities(self, G, U, suffix):
        ''' Calculate centralities in directed graph and add them as node attributes '''
        
        nx.set_node_attributes(G, nx.in_degree_centrality(U), 'in_degree_%s' % suffix)
        nx.set_node_attributes(G, nx.out_degree_centrality(U), 'out_degree_%s' % suffix)
        nx.set_node_attributes(G, nx.closeness_centrality(U.reverse(), u=None, distance=None, wf_improved=True), 'closeness_reverse_%s' % suffix)
        nx.set_node_attributes(G, nx.eigenvector_centrality(U, max_iter=1000, tol=1e-06, nstart=None, weight=None), 'eigenvector_%s' % suffix)
        nx.set_node_attributes(G, nx.pagerank(U, alpha=0.85, personalization=None, max_iter=1000, tol=1e-06, nstart=None, weight=None, dangling=None), 'pagerank_%s' % suffix)

        hub, authority = nx.hits(U, max_iter=1000, tol=1e-08, nstart=None, normalized=True)
        nx.set_node_attributes(G, hub, 'hub_%s' % suffix)
        nx.set_node_attributes(G, authority, 'authority_%s' % suffix)
        return G
    
    def calculate_undir_centralities(self, G, U, suffix):
        ''' Calculate centralities in undirected graph and add them as node attributes '''
        
        nx.set_node_attributes(G, nx.degree_centrality(U), 'degree_%s' % suffix)
        nx.set_node_attributes(G, nx.node_clique_number(U), 'clique_number_%s' % suffix)
        nx.set_node_attributes(G, nx.number_of_cliques(U), 'num_of_cliques_%s' % suffix)
        return G
    
    def extract_features(self, candidates_file, graph_file, node_features_file):
        
        '''
        Calculates node features based on graph centralities.
    
        Arguments:
            graph_file: Path to the csv file containing the graph edges
            node_attributes_file: Path to the csv file containing additional attributes for each node
            node_features_file: Output file
        '''
        
        with open(graph_file, 'rb') as inf:
            next(inf, '')
            G = nx.read_edgelist(inf, delimiter=self.separator, nodetype=str, encoding='utf-8', create_using=nx.DiGraph())
        
        node_metadata = pd.read_csv(candidates_file, sep=self.separator, encoding='utf-8', index_col=['title'], usecols= self.TEXT_COLUMNS)

        G.add_nodes_from(node_metadata.index.values)

        G = self.calculate_gen_centralities(G, G, 'd')
        G = self.calculate_dir_centralities(G, G, 'd')
        
        U = G.to_undirected()
        G = self.calculate_gen_centralities(G, U, 'u')
        G = self.calculate_undir_centralities(G, U, 'u')
        
        nodes = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
        nodes.index.name = 'title'
        
        features = pd.concat([nodes, node_metadata], axis=1, sort=False)
        features.index.name = 'title'
        features.reset_index().to_csv(node_features_file, sep=self.separator, encoding='utf-8', index=False)
    

class Classifier:
    '''
    The Classifier calculates the probability that one candidate is an academic discipline based on a trained classifier.
    
    Arguments:
        models_dir: Directory containing the trained models
        separator: Csv separator
    '''
    
    FEATURE_COLUMNS = ['closeness_d', 'closeness_reverse_d', 'betweenness_d', 'load_d', 'harmonic_d',
                       'in_degree_d', 'out_degree_d', 'eigenvector_d', 'pagerank_d', 'authority_d', 'hub_d',
                       'degree_u', 'closeness_u', 'betweenness_u', 'load_u', 'harmonic_u',
                       'clique_number_u', 'num_of_cliques_u',
                       'decision_tree_probability/confidence(svm)', 'linear_svm_1_probability/confidence(svm)',
                       'linear_svm_2_probability/confidence(svm)']
    OUTPUT_COLUMNS = ['title', 'class_predicted', 'class_probability']
    
    NODE_SCALER_MODEL = 'node-scaler.joblib'
    NODE_CLASSIFIER_MODEL = 'node-classifier.joblib'

    def __init__(self, models_dir, separator):
        self.models_dir = models_dir
        self.separator = separator
    
    def classify(self, node_features_file, disciplines_file):
        '''
        Calculates the probability that one candidate is an academic discipline based on a trained classifier.
    
        Args:
            node_features_file: Path to file containing node features
            disciplines_file: Output file
        '''
        X = pd.read_csv(node_features_file, sep=self.separator, encoding='utf-8')
        X_values = X[self.FEATURE_COLUMNS].values

        scaler = joblib.load(os.path.join(self.models_dir, self.NODE_SCALER_MODEL))
        classifier = joblib.load(os.path.join(self.models_dir, self.NODE_CLASSIFIER_MODEL))

        X_scaled = scaler.transform(X_values)
        y_predicted = classifier.predict(X_scaled)
        y_score = classifier.predict_proba(X_scaled)[:,1]

        X['class_predicted'] = y_predicted
        X['class_probability'] = y_score.reshape([y_score.shape[0],1])

        X = X[self.OUTPUT_COLUMNS]
        X = X.sort_values(by='class_probability', ascending=False)
        X.to_csv(disciplines_file, sep=self.separator, encoding='utf-8', index=False)
        
        
class NodeClassifier:
    
    GRAPH_FILE_SUFFIX = '-graph.csv'
    NODE_FEATURES_FILE_SUFFIX = '-node-features.csv'
    DISCIPLINES_FILE_SUFFIX = '-classified.csv'
    
    def __init__(self, models_dir, separator):
        self.graph_creator = GraphCreator(separator)
        self.feature_extractor = FeatureExtractor(separator)
        self.classifier = Classifier(models_dir, separator)
    
    def run(self, data_dir, candidates_file):
        graph_file = candidates_file.replace('.csv', self.GRAPH_FILE_SUFFIX)
        node_features_file = candidates_file.replace('.csv', self.NODE_FEATURES_FILE_SUFFIX)
        disciplines_file = candidates_file.replace('.csv', self.DISCIPLINES_FILE_SUFFIX)
        
        self.graph_creator.save_edges(data_dir, candidates_file, graph_file)
        self.feature_extractor.extract_features(candidates_file, graph_file, node_features_file)
        self.classifier.classify(node_features_file, disciplines_file)
