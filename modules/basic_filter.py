import os
import re
import json
import pandas as pd
from smart_open import smart_open

class Utilities:
    
    ''' Utility functions for filtering Wikipedia articles '''
    
    CONJUNCTIONS = ["for", "and", "nor", "but", "or", "yet", "so", "just", "either", "neither", "a", "an", "the", "by", "to", "vs", "vs.", "rather"]
    ADMIN_PREFIXES = ('List of', 'Timeline of', 'Lists of', 'Index of', 'Glossary of', 'Outline of', 'File:', 'Media:', 'Portal:', 'Template:', 'User:', 'Help:', 'Category:', 'Draft:', 'Talk:', 'Special:', 'Book:', 'Module:')
    ADMIN_KEYWORDS = ['Wikipedia', 'MediaWiki', 'WikiProject']
    PHRASES = ['may refer to:', 'may also refer to:']
    PATTERN_DIGIT_START = re.compile('[0-9]+[\s]in[\s][a-zA-z].+$')
    PATTERN_DIGIT_END = re.compile('^.+[a-zA-z][\s]in[\s][0-9]+$')
    
    def pre_extraction_filtering(self, title, text):
        to_be_classified = False;

        title = title if self.disambiguation_page(title) else re.sub('\s?\\(.+\\)\s?', '', title)
        title = title.replace('-', ' ')
        title = re.sub('\s{2,}', ' ', title)
        title = re.sub('^\s{1,}', '', title)
        title = re.sub('\s{1,}$', '', title)

        words = title.split(sep=' ')

        if len(words) == 1 and not self.may_refer_to(text) and not self.contains_digit(title) and not self.all_upper_case(words[0]) and not self.all_digits(words[0]):
            to_be_classified = True
        elif len(words) > 1 and not self.may_refer_to(text) and not self.disambiguation_page(title) and not self.contains_digit(title) and not self.admin_page(title) and not self.all_words_capitalized(words) and not self.topic_by_year(title):
            to_be_classified = True

        return to_be_classified
    
    def post_extraction_filtering(self, text):
        to_be_classified = True
        if any(keyword in text for keyword in self.PHRASES):
            to_be_classified = False

        return to_be_classified
    
    def all_words_capitalized(self, words):
        all_words_capitalised = True

        for word in words:
            if word[0].islower() and not self.conjunction(word):
                all_words_capitalised = False
                break

        return all_words_capitalised

    def conjunction(self, word):
        conjunction = False
        for c in self.CONJUNCTIONS:
            if word == c:
                conjunction = True
                break

        return conjunction


    def all_upper_case(self, word):
        all_upper_case = True
        for i in range(0, len(word)):
            if word[i].islower():
                all_upper_case = False
                break;

        return all_upper_case;


    def all_digits(self, word):
        all_digits = True
        for i in range(0, len(word)):
            if not word[i].isnumeric():
                all_digits = False
                break

        return all_digits


    def admin_page(self, title):
        admin_page = False
        if title.startswith(self.ADMIN_PREFIXES) or any(keyword in title for keyword in self.ADMIN_KEYWORDS):
            admin_page = True

        return admin_page


    def disambiguation_page(self, title):
        disambiguation_page = False
        if 'disambiguation' in title:
            disambiguation_page = True

        return disambiguation_page;


    def topic_by_year(self, title):
        topic_by_year = False
        if self.PATTERN_DIGIT_START.fullmatch(title) is not None or self.PATTERN_DIGIT_END.fullmatch(title) is not None:
            topic_by_year = True

        return topic_by_year


    def contains_digit(self, title):
        contains_digit = False
        if re.search('.*[0-9]+.*', title) is not None:
            contains_digit = True

        return contains_digit
    

    def may_refer_to(self, text):
        may_refer_to = False
        if not '.' in text.replace('.', '', 1) and text.endswith(tuple(self.PHRASES)):
            may_refer_to = True

        return may_refer_to
    

class LeadSectionExcerptExtractor:
    
    def __init__(self, word_limit_mean = 50, word_limit_std = 25):
        self.word_limit_mean = word_limit_mean
        self.word_limit_std = word_limit_std
        self.word_limit_min = word_limit_mean - word_limit_std
        self.word_limit_max = word_limit_mean + word_limit_std

    def append_paragraphs(self, paragraphs):
        text = ""
        start_chars = ('*', ':', ';', '#')
        end_chars = ('.', ':', ',', ';')
        for paragraph in paragraphs:
            if paragraph.startswith(start_chars):
                paragraph = re.sub('^\*', '', paragraph)
                paragraph = re.sub('^:', '', paragraph)
                paragraph = re.sub('^;', '', paragraph)
                paragraph = re.sub('^#', '', paragraph)
                paragraph = paragraph + ';'
            if text.endswith(end_chars):
                text = text + ' ' + paragraph
            else:
                text = text + paragraph
        return text
        
    def split(self, appended_paragraphs):
        sentences = []
        sentences_and_separators = re.split('(\.|;)',appended_paragraphs)
        if len(sentences_and_separators) > 1:
            for i in range(0, len(sentences_and_separators)-1, 2):
                sentences.append(sentences_and_separators[i] + sentences_and_separators[i+1])
            sentences.append(sentences_and_separators[len(sentences_and_separators)-1])
        else:
            sentences.append(sentences_and_separators[0])
        return sentences
        
    def filter_sentences(self, sentences):
        index = 0
        for sentence in sentences:
            if sentence.startswith('For'):
                if re.match('For\s.+see\s.+\.', sentence):
                    index += 1
                else:
                    break
            else:
                break
        return sentences[index:]
        
    def append_sentences(self, sentences):
        text = ""
        current_length = 0
        for sentence in sentences:
            sentence_length = sentence.count(" ") + 1
            new_length = current_length + sentence_length
            if current_length < self.word_limit_mean:
                if new_length <= self.word_limit_max:
                    text = sentence if text == "" else text + sentence
                    current_length = new_length
                elif current_length > self.word_limit_min:
                    break
                else:
                    text = sentence if text == "" else text + sentence
                    break
            else:
                break
        return text
        
        
    def extract(self, page_lead_section, title):
        page_lead_section = re.sub("\s{2,}", " ", page_lead_section)
        paragraphs = page_lead_section.split('\n')
        
        nonempty_paragraph = re.compile("^.*[A-Za-z0-9].*$")
        paragraphs = list(filter(nonempty_paragraph.match, paragraphs))
        if len(paragraphs)==0:
            paragraphs.append('')
        
        appended_paragraphs = self.append_paragraphs(paragraphs)
        sentences = self.split(appended_paragraphs)
        sentences = self.filter_sentences(sentences)
        if len(sentences) == 1 and '.' not in appended_paragraphs and appended_paragraphs.count(" ") > self.word_limit_max:
            lead_section = ""
        else:
            lead_section = self.append_sentences(sentences)

        # additional corrections
        lead_section = re.sub("'{2,}", "", lead_section)
        lead_section = re.sub("\([^A-Za-z0-9]*\)", " ", lead_section)
        lead_section = re.sub("_{1,}NOTOC_{1,}", "", lead_section)
        lead_section = re.sub("^\{\|[\s]*", "", lead_section)
        lead_section = re.sub("\s{2,}", " ", lead_section)
        lead_section = lead_section.strip()
        
        if lead_section.startswith(','):
            lead_section = title + lead_section
        if lead_section.startswith('" "'):
            lead_section = title + lead_section.replace('" "', '')
                    
        return lead_section
    
    
class BasicFilter:
    
    '''
    The BasicFilter filters Wikipedia articles that do not follow the patterns common for academic disciplines titles 
    and extracts short representative subsection from the article’s lead section.
    
    Arguments:
        separator: Csv separator
    '''
    BATCH_SIZE = 10000
    ARCHIVE_SUFFIX = '.gz'
    FILTERED_FILE_SUFFIX = '-filtered.csv'
    
    def __init__(self, separator):
        self.separator = separator
        self.Utilities = Utilities()
        self.LeadSectionExcerptExtractor = LeadSectionExcerptExtractor(word_limit_mean=50, word_limit_std=25)
        
    def write(self, data, path, sep, append=True, header=True, encoding='utf-8'):
        data.to_csv(path, sep=sep, encoding=encoding, index=False, mode='a', header=header)
    
    def filter_(self, json_file, output_file):
        retained_elements = 0
        selected_data = pd.DataFrame(columns=['title', 'text'])
        
        for line in smart_open(json_file):
            article = json.loads(line)
            title = article['title']
            text = ''

            for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                if section_title == 'Introduction':
                    text = section_text
                
            if text != '' and self.Utilities.pre_extraction_filtering(title, text):
                text = self.LeadSectionExcerptExtractor.extract(text, title)
                
                if text != '' and self.Utilities.post_extraction_filtering(text):
                    selected_data = selected_data.append({'title':title,
                                                          'text':title + '. ' + text},
                                                         ignore_index=True)
                    retained_elements += 1
                    if retained_elements % self.BATCH_SIZE == 0:
                        self.write(selected_data, output_file, sep=self.separator, header=(retained_elements<=self.BATCH_SIZE))
                        selected_data = pd.DataFrame(columns=['title', 'text'])
        
        self.write(selected_data, output_file, sep=self.separator, header=(retained_elements<=self.BATCH_SIZE))
        
    def run(self, directory):
        json_files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(self.ARCHIVE_SUFFIX)]
        for json_file in json_files:
            self.filter_(json_file, json_file.replace(self.ARCHIVE_SUFFIX, self.FILTERED_FILE_SUFFIX))
