import os
import io
import re
import bz2
import gensim.scripts.segment_wiki as segment_wiki


class Preprocessor:
    
    '''
    The Preprocessor class provides optional preprocessing before the metadata and text extraction. 
    The code is provided for completeness only, as it was applied to get the results reported in the paper, 
    but may not be needed or should be modified depending on the Wikipedia exports and Gensim version.
    '''
    
    BATCH_SIZE = 10000
    
    REGEX_1 = "\[\[([fF]ile:|[iI]mage:)[^\[\]]+(\[\[[^\[\]]+\]\]([^\[\]]+)?)*\]\]"
    REGEX_2 = "<text xml:space=\"preserve\">(\{\{[^\n]*\}\})?\s*[\n]+:{1,3}\s{0,}''[^\n]+</text>$"
    REGEX_3 = '<text xml:space=\"preserve\">\n</text>'
    REGEX_4 = "<text xml:space=\"preserve\">(\{\{[^\n]*\}\})?\s*[\n]+:{1,3}\s{0,}''[^\n]+(?!</text>)$"
    REGEX_5 = '<text xml:space=\"preserve\">\n'
    REGEX_6 = "\}\}[\n]+:\s{0,}''[^\n]+('')?</text>$"
    REGEX_7 = '\}\}\n</text>'
    REGEX_8 = "\}\}[\n]+:\s{0,}''[^\n]+('')?$"
    REGEX_9 = '\}\}\n'
    REGEX_10 = "^.{0,4}border=.*"
    REGEX_11 = "\n.{0,4}border=.*"
    REGEX_12 = "<text xml:space=\"preserve\">.{0,4}border=.*"
    REGEX_13 = "<text xml:space=\"preserve\">"
    REGEX_14 = "&lt;ref((?!&gt;).)*&gt;((?!&lt;/ref&gt;).)*&lt;/ref&gt;"
    REGEX_15 = "&lt;ref((?!&gt;).)*((?!&lt;ref).)*\s*/&gt;"
    REGEX_16 = "\[\[([fF]ile:|[iI]mage:)[^\[\]]+(\[\[[^\[\]]+\]\]([^\[\]]+)?)*\]\]"
    
    def delete(self, file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print("Error: %s file not found" % file_path)
        
    def decompress(self, compressed_input_file_path, new_file_path):
        with open(new_file_path, 'wb') as new_file, bz2.BZ2File(compressed_input_file_path, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                new_file.write(data)
                
    def compress(self, new_compressed_file_path, file_path):
        with open(file_path, 'rb') as new_file, bz2.BZ2File(new_compressed_file_path, 'wb') as file:
            for data in iter(lambda : new_file.read(100 * 1024), b''):
                file.write(data)
        
    def correct(self, page_lines):
        page_lines = re.sub(self.REGEX_1, "", page_lines)
        page_lines = re.sub(re.compile(self.REGEX_2, re.MULTILINE), self.REGEX_3, page_lines)
        page_lines = re.sub(re.compile(self.REGEX_4, re.MULTILINE), self.REGEX_5, page_lines)
        page_lines = re.sub(re.compile(self.REGEX_6, re.MULTILINE), self.REGEX_7, page_lines)
        page_lines = re.sub(re.compile(self.REGEX_8, re.MULTILINE), self.REGEX_9, page_lines)        
        page_lines = re.sub(re.compile(self.REGEX_10, re.MULTILINE), "", page_lines)
        page_lines = re.sub(self.REGEX_11, "\n", page_lines)
        page_lines = re.sub(self.REGEX_12, self.REGEX_13, page_lines)
        page_lines = re.sub(self.REGEX_14, "", page_lines)
        page_lines = re.sub(self.REGEX_15, "", page_lines)        
        page_lines = re.sub(self.REGEX_16, "", page_lines)
        
        return page_lines
        
    def parse(self, compressed_input_file_path):
        input_file_path = compressed_input_file_path.replace(".bz2", "")
        self.decompress(compressed_input_file_path, input_file_path)
            
        output_file_path = input_file_path.replace(".xml", "-preprocessed-by-ADD.xml")
        with io.open(input_file_path, mode="r", encoding="utf-8") as file, io.open(output_file_path, 'a', encoding='utf8') as fd:
            in_page = False
            page_lines = ""
            i=0
            for line in file:
                if '<page>' in line:
                    in_page = True
                    page_lines = ""
                if in_page and '</page>' in line:
                    in_page = False
                    page_lines = page_lines + line
                    line = self.correct(page_lines)
                    
                if not in_page:
                    fd.write(line)
                    i+=1
                    if i % self.BATCH_SIZE == 0:
                        fd.flush()
                else:
                    page_lines = page_lines + line
            fd.flush()
                    
        new_compressed_file_path = output_file_path + ".bz2"
        self.compress(new_compressed_file_path, output_file_path)
            
        self.delete(input_file_path)
        self.delete(output_file_path)
        
        return new_compressed_file_path
            
class Extractor:   
    '''
    The Extractor class reads Wikipedia XML export files and produces JSON files containing extracted metadata.
    
    Arguments:
        preprocess: boolean, whether to apply optional preprocessing before metadata and text extraction 
        (provided for completeness, as it was applied to get the results reported in the paper, 
        but may not be needed for other Wikipedia exports)
    '''
    FILE_SUFFIX ='.bz2'
    
    def __init__(self, preprocess=False):
        self.preprocess = preprocess
        self.preprocessor = None
        if self.preprocess:
            self.preprocessor = Preprocessor()
            
    def run(self, directory):
        '''
        Reads Wikipedia XML export files and produces JSON files containing extracted metadata.
    
        Arguments:
            directory: string, directory containing the Wikipedia XML export files to be processed
        '''
        files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(self.FILE_SUFFIX)]
        for input_file in files:
            if self.preprocess:
                input_file = self.preprocessor.parse(input_file)
            output_file = input_file.replace('.xml', '.json').replace('.bz2', '.gz')
            segment_wiki.segment_and_write_all_articles(input_file, output_file, min_article_character=200, include_interlinks=True)
            if self.preprocess:
                self.preprocessor.delete(input_file)
