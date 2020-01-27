import os
import gensim.scripts.segment_wiki as segment_wiki


class Extractor:   
    '''
    The Extractor class reads Wikipedia XML export files and produces JSON files containing extracted metadata.
    '''
    FILE_SUFFIX ='.bz2'
    
    def __init__(self, preprocess=False):
        self.preprocess = preprocess
            
    def run(self, directory):
        '''
        Reads Wikipedia XML export files and produces JSON files containing extracted metadata.
    
        Arguments:
            directory: string, directory containing the Wikipedia XML export files to be processed
        '''
        files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(self.FILE_SUFFIX)]
        for input_file in files:
            output_file = input_file.replace('.xml', '.json').replace('.bz2', '.gz')
            segment_wiki.segment_and_write_all_articles(input_file, output_file, min_article_character=200, include_interlinks=True)
