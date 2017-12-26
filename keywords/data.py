import numpy as np
import unicodedata
import string
import wiki
import sys

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))

def filterText(text):
	global tbl
	if type(text) == str:
		text = text.translate(None, string.punctuation)
	elif type(text)	== unicode:
		text = text.translate(tbl)
	return text.split()

def getVocabulary(cmap):
	vocabulary = []
	for title in cmap.keys():
		summary = filterText(cmap[title][0])
		vocabulary += list(set(summary))	
	vocabulary = list(set(vocabulary))		
	vocabulary.sort()
	return vocabulary

def encodeText(text, vocabulary, max_input_size):
	word_index_map = dict([(word, i) for i, word in enumerate(vocabulary)])
	vector = np.zeros((max_input_size, len(vocabulary)))
	for i, word in enumerate(text):
		vector[i,word_index_map[word]] = 1
	return vector

def getData(cmap,vocabulary):
	input_raw_text, ouput_raw_text = [], []
	for title in cmap.keys():
		summary = filterText(cmap[title][0])
		keyphrases = cmap[title][1]	
		for keyphrase in keyphrases:
			keyphrase = filterText(keyphrase)
			input_raw_text.append(summary)   
			ouput_raw_text.append(keyphrase) 
	return input_raw_text, ouput_raw_text

def run():
	cmap = wiki.getCmap()
	vocabulary = getVocabulary(cmap)
	vocab_size =  len(vocabulary)
	input_raw_text, output_raw_text = getData(cmap, vocabulary)
	max_input_size = max([len(data) for data in input_raw_text])
	max_output_size = max([len(data) for data in output_raw_text])
	input_encoded_text = [encodeText(summary, vocabulary, max_input_size) for summary in input_raw_text]
	ouput_encoded_text = [encodeText(keyphrase, vocabulary, max_output_size) for keyphrase in output_raw_text]
	shifted_ouput_encoded_text = [encodeText(keyphrase[1:], vocabulary, max_output_size) for keyphrase in output_raw_text]

