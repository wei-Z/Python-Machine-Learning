import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
from sklearn.linear_model import LogisticRegression

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/') # get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-') # get tokens after splitting by dash
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split('.') # get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens # list union
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))   # remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com') # removing .com since it occurs a lot of times and it should not be included in our features
    return allTokens
    
allurls = 'C:\Users\Wei\Desktop\\Using machine learning to detect malicious URLs\\data\data.csv'
allurlscsv = pd.read_csv(allurls, ',', error_bad_lines=False) # reading file
allurlsdata = pd.DataFrame(allurlscsv) # converting to a dataframe

allurlsdata = np.array(allurlsdata) # converting it into an array
random.shuffle(allurlsdata) # shuffling

y = [d[1] for d in allurlsdata] # all labels
corpus = [d[0] for d in allurlsdata] #all urls corresponding to a label(either good or bad)
# get a vector for each url but use our customized tokenizer
vectorizer = TfidfVectorizer(tokenizer=getTokens) 
#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer followed by TfidfTransformer.

X = vectorizer.fit_transform(corpus) # get the X vector

# #split into training and testing set 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Using Logistic Regression
lgs = LogisticRegression()
lgs.fit(X_train, y_train)
print lgs.score(X_test, y_test) # printing the score

# #checking some random URLs. The results come out to be expected. 
#The first two are okay and the last four are malicious/phishing/bad
X_predict =  ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
X_predict = vectorizer.transform(X_predict)
y_Predict = lgs.predict(X_predict)
print y_Predict	#printing predicted values






