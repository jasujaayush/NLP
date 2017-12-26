import wikipedia
import pickle
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings() #disabling wikipedia package warnings

cmap = {}
concept_queue = ['Steve Jobs']
parsed_concepts = []
total_pages = 1000
def wikiTraversal(title):
	concepts = []	
	print "Trying to parse page : ", title	
	try:
		page = wikipedia.page(title)
		for link in page.links:
			if link.lower() in page.summary.lower():
				concepts.append(link)
		cmap[title] = (page.summary, concepts)
	except:		
		print "passing an exception"
		pass	
	return concepts

def getCmap():
	with open('cmap.pkl', 'rb') as handle:
		data = pickle.load(handle)
	print "Returning data size : ", len(data)
	return data

def saveCmap():
	i = 0
	while i < len(concept_queue):
		if len(cmap) > total_pages:
			break
		concept = concept_queue[i]
		if concept not in parsed_concepts:
			concept_queue = concept_queue + wikiTraversal(concept)
			parsed_concepts.append(concept)
		i += 1
	with open('cmap.pkl', 'wb') as handle:
		pickle.dump(cmap, handle, protocol=pickle.HIGHEST_PROTOCOL)