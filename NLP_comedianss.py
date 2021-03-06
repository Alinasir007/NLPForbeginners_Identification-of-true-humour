import nltk
import requests
import pickle
from bs4 import BeautifulSoup

def url_to_transcript(url):
    page= requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text= [p.text for p in soup.find(class_= "post-content").find_all('p')]
    print(url)
    return text

urls= ['https://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
      'https://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
      'https://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
      'https://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
      'https://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
      'https://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
      'https://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
      'https://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
      'https://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
      'https://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
      'https://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
      'https://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

comedians=['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

transcripts= [url_to_transcript(u) for u in urls]

for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)

data = {}
for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c]= pickle.load(file)

data.keys()
dict_keys= (['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe'])
print(data['louis'][:2])

print(next(iter(data.keys())))
print(next(iter(data.values())))

#compile data in one string
def combine_text(list_of_text):
    combined_text= ' '.join(list_of_text)
    return combined_text

data_combined= {key: [combine_text(value)] for (key, value) in data.items()}

import pandas as pd
pd.set_option('max_colwidth', 150)

data_df= pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns= ['transcript']
data_df= data_df.sort_index()
print(data_df)

print(data_df.transcript.loc['ali'])

#Cleaning
import re
import string
def clean_text_round1(text):
    #text= str(text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[''"".]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = text.lower()
    text = re.sub('\w*\d\w*', '', text)
    return text

round1= lambda x: clean_text_round1(x)


data_clean= pd.DataFrame(data_df.transcript.apply(round1))
print((data_clean))

full_names= ['Ali wong', 'Anthony jeselnik', 'Bill burr', 'Bo burnham', 'Dave Chappelle', 'Hasan Minhanj', 'Jim jefferies', 'Joe rogan', 'john mulaney', 'Louis c.k', 'Mike birbiglia', 'Ricky gervais']

data_df.to_pickle("corpus.pkl")

from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(stop_words= 'english')
data_cv= cv.fit_transform(data_clean.transcript)
data_dtm= pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index= data_clean.index
print(data_dtm)

data_dtm.to_pickle("dtm.pkl")
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open('cv.pkl', 'wb'))


datanew= pd.read_pickle('dtm.pkl')
datanew= datanew.transpose()
print(datanew.head())

#find top 30 words
top_dict= {}
for c in datanew.columns:
    top= datanew[c].sort_values(ascending= False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

print(top_dict)

#top 15 words for every comedian
for comedian, top_words in top_dict.items():
    print(comedian)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('----')

from collections import Counter
words= []
for comedian in datanew.columns:
    top= [word for (word, count) in top_dict[comedian]]
    for t in top:
        words.append(t)

print(words)

print(Counter(words).most_common())

add_stop_words= [word for word, count in Counter(words).most_common() if count > 6 ]
print(add_stop_words)

from sklearn.feature_extraction import text

#data clean
data_clean = pd.read_pickle('data_clean.pkl')
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

#re-create documentvmatrix
cv= CountVectorizer(stop_words= stop_words)
data_cv= cv.fit_transform(data_clean.transcript)
data_stop= pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index= data_clean.index

#pickle it for later
pickle.dump(cv, open('cv_stop.pkl', 'wb'))
data_stop.to_pickle('dtm_stop.pkl')


#word clouds
from wordcloud import WordCloud
wc= WordCloud(stopwords= stop_words, background_color='white', colormap='Dark2', max_font_size= 150, random_state= 42)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= [16,6]
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
for index, comedian in enumerate(datanew.columns):
    wc.generate(data_clean.transcript[comedian])

    plt.subplot(3,4, index + 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(full_names[index])

plt.show()

#Analysis
# Find the number of unique words that each comedian uses

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for comedian in datanew.columns:
    uniques = datanew[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
print(data_unique_sort)

# Calculate the words per minute of each comedian

# Find the total number of words that a comedian uses
total_list = []
for comedian in datanew.columns:
    totals = sum(datanew[comedian])
    total_list.append(totals)

# Comedy special run times from IMDB, in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')
data_wpm_sort

# Let's plot our findings
import numpy as np

y_pos = np.arange(len(data_words))

plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.comedian)
plt.title('Number of Unique Words', fontsize=20)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.comedian)
plt.title('Number of Words Per Minute', fontsize=20)

plt.tight_layout()
plt.show()

# revisit profanity: most common words again.
Counter(words).most_common()

# Let's isolate just these bad words
data_bad_words = datanew.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']
data_profanity

# Let's create a scatter plot of our findings
plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x + 1.5, y + 0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155)

plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()