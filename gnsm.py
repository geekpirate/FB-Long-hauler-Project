# from gensim.models import Word2Vec
import re
import string

from idna import unichr
from joblib.numpy_pickle_utils import xrange
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
import nltk
import csv, os
import unicodedata, re , sys
from textblob import TextBlob
# uncomment below lines when the code is compiled for the first time
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
directory = os.fsencode("/Users/sairo/Documents/ark/Project/DE/Comments/Survior Corps_Comments")
for file in os.listdir(directory):
	#print(filename)
	filename = os.fsdecode(file)
	delim = "__________________________________________________________________________\n"
	sentences = []
	out = []
	hashtags = []
	temp = 0
	wordcount = 0
	post = ""
	lemmatizer=WordNetLemmatizer()
	s = set(stopwords.words('english'))
	count = 0
	#csvfile = io.open('features.csv', 'w')
	# with open("features.csv", 'w') as csvfile:
		# creating a csv writer object
	fieldnames = ['Post', 'HighFrequencyWords', 'WordCount']
	# csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
	# csvwriter.writeheader()
		# csvwriter.writerow(fieldnames)
	with open("/Users/sairo/Documents/ark/Project/DE/Comments/Survior Corps_Comments/"+filename, encoding="utf8") as fp:

		lines = fp.readlines()
		for i in range(0, len(lines)):
			count += 1
			if temp == 0:
				temp+=1
				continue
			if lines[i] == delim:
				temp = 0
				frequency_dist = nltk.FreqDist(out)
				highFrequencyWords = sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:5]
				# print(sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:5])
				# print(wordcount)
				# print(hashtags)
				# print(len(hashtags))
				#post = remove_control_chars(post)
				filter(lambda x: x in string.printable, post)
				# encoded_string = post.encode("ascii", "ignore")
				# post = encoded_string.decode()
				# post.replace()
				#csvwriter.writerow({'Post': post, 'HighFrequencyWords': highFrequencyWords, 'WordCount': wordcount})
				hashtags = []
				post = ""
				wordcount = 0
				#break
				# continue
			if lines[i] != delim:
				post += lines[i]
			string.punctuation = string.punctuation + '’' + '‘'
			text_tokens = word_tokenize(lines[i])
			wordcount += len(text_tokens)
			out = [word for word in text_tokens if not word in s]
			out = [word for word in out if not word in string.punctuation]
			out = [lemmatizer.lemmatize(word) for word in out]
			out = [word for word in out if not (nltk.pos_tag(word)[0][1] == "PRP")]
			#hashtags = hashtags + [text_tokens[i] + text_tokens[i + 1] for i in range(len(text_tokens)) if
						#		   text_tokens[i].startswith('#')]

			sentences.append(out)
			#print(lines[i])
		#print(sentences)

	model = Word2Vec(sentences, min_count=1)
	# fit a 2d PCA model to the vectors
	X = model[model.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection
	pyplot.scatter(result[:, 0], result[:, 1])
	words = list(model.wv.vocab)
	print("done")
	# out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
	# out_m = io.open('metadata.tsv', 'w', encoding='utf-8')C:\Users\sairo\Documents\ark\Project\DE\Features
	for i, word in enumerate(words):
		out_v = io.open("/Users/sairo/Documents/ark/Project/DE/Features/Comment_features/Survior Corps_Comments/"+filename.split()[0]+"Post_"+str(i)+"_vectors.tsv", 'w', encoding='utf-8')
		out_m = io.open("/Users/sairo/Documents/ark/Project/DE/Features/Comment_features/Survior Corps_Comments/"+filename.split()[0]+"Post_"+str(i)+"_metadata.tsv", 'w', encoding='utf-8')
		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
		if i == 0: continue  # skip 0, it's padding.
		vec = model[word]
		out_v.write('\t'.join([str(x) for x in vec]) + "\n")
		out_m.write(word + "\n")
		out_v.close()
		out_m.close()
	#csvfile.close()
	fp.close()
#pyplot.show()
print("Completed")