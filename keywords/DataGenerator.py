import numpy as np
import unicodedata
import string
import random
import wiki
import sys

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, cmap, batch_size = 32, shuffle = True):
      'Initialization'
      self.cmap = cmap
      print "getting vocabulary"
      vocabulary = self.getVocabulary()
      print "setting word index map"
      self.word_index_map = dict([(word, i) for i, word in enumerate(vocabulary)])
      print "getting raw text"
      input_raw_text, output_raw_text = self.getRawData()
      self.max_input_size = max([len(data) for data in input_raw_text])
      self.max_output_size = max([len(data) for data in output_raw_text])
      self.vocab_size = len(vocabulary)
      self.batch_size = batch_size
      self.shuffle = shuffle

  def filterText(self, text):
    import sys
    tbl = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))
    if type(text) == str:
      text = text.translate(None, string.punctuation)
    elif type(text) == unicode:
      text = text.translate(tbl)
    return text.split()

  def getVocabulary(self):
    vocabulary = []
    for title in self.cmap.keys():
      summary = self.filterText(self.cmap[title][0])
      vocabulary += list(set(summary))  
      vocabulary = list(set(vocabulary))    
    vocabulary.sort()
    return vocabulary

  def encodeText(self, text):
    vector = np.zeros((self.max_input_size, self.vocab_size))
    for i, word in enumerate(text):
      if self.word_index_map.has_key(word):
        vector[i,self.word_index_map[word]] = 1
      elif self.word_index_map.has_key(word.lower()):
        vector[i,self.word_index_map[word.lower()]] = 1
      else:
        pass  
    return vector

  def getRawData(self):
    input_raw_text, ouput_raw_text = [], []
    for title in self.cmap.keys():
      summary = self.filterText(self.cmap[title][0])
      input_raw_text.append(summary)   
      keyphrases = self.cmap[title][1] 
      for keyphrase in keyphrases:
        keyphrase = self.filterText(keyphrase)
        ouput_raw_text.append(keyphrase) 
    return input_raw_text, ouput_raw_text

  def generate(self):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          #print "here"
          data_pairs = []
          for i in range(self.batch_size):
            title = random.choice(self.cmap.keys())
            summary = self.filterText(self.cmap[title][0])
            keyphrase = self.filterText(random.choice(self.cmap[title][1]))
            data_pairs.append((summary, keyphrase))
          input_encoded_text = np.array([self.encodeText(summary) for summary, keyphrase in data_pairs])
          ouput_encoded_text = np.array([self.encodeText(keyphrase) for summary, keyphrase in data_pairs])
          shifted_ouput_encoded_text = np.array([self.encodeText(keyphrase) for summary, keyphrase in data_pairs])
          yield [input_encoded_text, ouput_encoded_text], shifted_ouput_encoded_text