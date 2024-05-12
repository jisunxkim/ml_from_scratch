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


if __name__ == "__main__":
    # download data
    url= "https://raw.githubusercontent.com/UmarRajpoot/Identify-Spam-Email-ML-Model/master/mail_data.csv"
    file_path="../tmp/"
    file_name="mail_spam_ham_data.csv"
    file = os.path.join(file_path, file_name)

    # load data
    data = pd.read_csv(file)
    data = data.values
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

    # sklearn 
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer


    vectorizer = CountVectorizer(ngram_range=(1,2))
    train_vector = vectorizer.fit_transform(train_data[:3, 1])
    # temp = vectorizer.transform(['no no', 'no no no will'])
    # print(temp.shape)
    # print(temp)
    print(vectorizer.vocabulary_)
    temp = vectorizer.transform(['no will check check'])
    print(type(temp))
    print(temp.shape)
    # print()