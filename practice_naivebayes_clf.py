from util_download_data_from_url import download_data_from_url
import os 
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

class NaiveBayesClassifier:
    """
    Naive bayes Algorithm
    A: no-buy, buy
    B1...Bn: feature
    Calculate: P(A_buy|B1...Bn) vs P(A_no_buy: B1...Bn)
    Select a label with larger conditional probability
    
    How to calculate the probability P(A|B):
    P(A|B) = [P(B|A) * P(A)] / P(B)

    For text features:
    B0 ~ B1: words and their count for probability
    """
    def __init__(self):
        self.label_prob = {} # P(A)
        self.feature_prob = {} # P(B)
        
    def train(self, x:str, y:str) -> None:
        """
        x: sentence, (m, 1)
        y: label, (m, 1)
        m: sample size
        """

def make_token(sentence:str, stopwords=None):
    sentence = sentence.lower().strip()
    # if not stopwords:
    #     sentence = ''.join([c for c in sentence 
    #                         if c.isalpha() or c == ' '])
    # else:
    #     sentence = ' '.join(word for word in sentence.split() 
    #                         if word not in stopwords)

    return sentence.split()

def make_vocaburary(token:list, voca:list=None):
    if not voca:
        voca = {}

    for word in token:
        if word not in voca:
            voca.append(word)
    return voca

if __name__ == "__main__":
    # download data
    # url= "https://raw.githubusercontent.com/UmarRajpoot/Identify-Spam-Email-ML-Model/master/mail_data.csv"
    # url = "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv"
    file_path = "/Users/q1460293/projects/tmp/"
    file_name="mail_spam.csv"
    file = os.path.join(file_path, file_name)
    # download_data_from_url(url, file_path, file_name)
    
    # load data
    data = pd.read_csv(file, encoding='latin1')
    data = data.iloc[:, :2].values
    print(data.shape)
    print(data[0])
    print("labels: ", np.unique(data[:,0]))

    # train test split
    np.random.seed(1)
    test_size = 0.2
    dev_size = 0.2
    np.random.shuffle(data)
    m = data.shape[0]
    test_data = data[:int(m*test_size)]
    dev_data = data[int(m*test_size):int(m*test_size)+int(m*dev_size)]
    train_data = data[int(m*test_size)+int(m*dev_size):]
    print(f"data size of train, dev and test:\n \
          {train_data.shape[0]}, {dev_data.shape[0]}, {test_data.shape[0]}")

    X = train_data[:,1]
    y = train_data[:,0]
    y = np.where(y == 'spam', 1, 0)
    labels = np.unique(y)
    
    
    X_train = [
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],
    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],
    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L']]

    X = [[str(c) for c in r] for r in X_train]
    X = [' '.join(item) for item in X]
    y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
     
    # make voca
    voca = {}
    tokens = []
    stopwords = ',.\'\"'
    for sentence in X:
        token = make_token(sentence, stopwords=None)
        tokens.append(token)
    i = 0
    for doc in tokens:
        for word in doc:
            if word not in voca:
                voca[word] = i 
                i += 1
    
    # calculate probabilities
    train_vector = np.ones((len(labels), len(voca)))
    for doc_idx, label in enumerate(y):
        for word in tokens[doc_idx]:
            if word in voca:
                train_vector[label, voca[word]] += 1     
    
    # p(feature|class)
    prob_features_given_class = train_vector / np.sum(train_vector, axis=1, keepdims=True)
    
    prob_labels = np.zeros(len(labels)).reshape(2,-1)
    
    for label in y:
        prob_labels[label] += 1

    prob_labels = prob_labels / np.sum(prob_labels, keepdims=True)
    
    temp_X = X[:30]
    temp_y = y[:30]
    # predict 
    for i in range(30):
        # X = train_data[i, 1]
        # y = train_data[i, 0]
        # y = np.where(y == 'spam', 1, 0)
        
        X= temp_X[i]
        y=temp_y[i]
        
        prob_pred = np.zeros((len(labels), 1))
        
        # token
        token = make_token(X)
        # feature word index
        X_features_idx = [voca[word] for word in token if word in voca]
        
        # p(label|features) = p(features|label)*p(label)/p(all features)
        pred_labels = np.zeros((1, len(labels)))
        for label in labels:
            # pred_labels[0, label] = np.prod(prob_features_given_class[label, X_features_idx]) * prob_labels[label] / np.prod(prob_features)
            # print(X_features_idx)
            pred_labels[0, label] = np.prod(prob_features_given_class[label, X_features_idx]) * prob_labels[label]
        
        # normalize
        pred_labels = pred_labels / np.sum(pred_labels)
        # print(pred_labels)
        print(np.argmax(pred_labels), temp_y[i])
        
        
        
        
        
        
    
    
       
    
        
    # sklearn 
    # word count vector
    # from sklearn.naive_bayes import MultinomialNB
    # from sklearn.feature_extraction.text import CountVectorizer

    # vectorizer = CountVectorizer(ngram_range=(1,1))
    # train_vector = vectorizer.fit_transform(train_data[:3, 1])
    # print(vectorizer.vocabulary_)
    # print(len(vectorizer.vocaburary_.keys()))
    # print(train_data[:3, 1])
    # temp = vectorizer.transform(['no will check check'])
    # print(temp.toarray())
    # print(temp.shape)
    