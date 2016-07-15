#Chapter 9 Embedding a Machine Learning Model into a Web Application
#Serializing fitted scikit-learn estimators

###############from Chapter 8###################
from nltk.corpus import stopwords

stop = stopwords.words('english')
############################################
import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
################vectorizer.py#####################
from sklearn.feature_exztraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                         + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
    
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
###############################################
import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
'''
After we have successfully loaded the vectorizer and unpickled the classifier, we can
now use these objects to pre-process document samples and make predictions about
their sentiment:
'''
import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print 'Prediction: %s\nProbability: %.2f%%' %\
        (label[clf.predict(X)[0]],
        np.max(clf.predict_proba(X))*100)

# Setting up a SQLite database for data storage
import sqlite3
import os
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db' \
                ' (review TEXT, sentiment INTEGER, data TEXT)')
example1 = 'I love this movie' 
c.execute("INSERT INTO review_db" \
                "(review, sentiment, date) VALUES'\
                '(?, ?, DATETIME('now'))", (example1, 1))
example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db' \
                '(?, ?, DATETIME('now'))", (example2, 0))
conn.commit()
conn.close()

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date" \
                " BETWEEN '2015-01-01 00:00:00' AND DATETIME('now')" )
results = c.fetchall()
conn.close()
print results

