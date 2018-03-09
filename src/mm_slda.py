# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
from sklearn.metrics import accuracy_score
import sys
import time
from svm import SVM
import svmpy
import vb

class SLDA:
    def __init__(self, n_topic=10, n_iter=20, alpha=0.1, beta=0.01):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, curpus, sample_y):
        word_indexes = []
        word_counts = []
        for row_curpus in curpus:
            row_indexes = []
            row_counts = []
            for w_i, w_c in row_curpus:
                row_indexes.append(w_i)
                row_counts.append(w_c)
            word_indexes.append(row_indexes)
            word_counts.append(row_counts)
        
        n_documents = len(word_indexes)    
        
        max_index = 0
        for d in range(n_documents):
            document_max = np.max(word_indexes[d])
            if max_index < document_max:
                max_index = document_max
                
        n_word_types = max_index + 1
        
        theta = np.random.uniform(size=(n_documents, self.n_topic))
        old_theta = np.copy(theta)
        phi = np.random.uniform(size=(self.n_topic, n_word_types))
        svm_alpha = np.random.uniform(size=n_documents) + 1e-3
        reg_weights = np.random.normal(size=self.n_topic)
        
        for n in range(self.n_iter):
            sum_phi = []
            for k in range(self.n_topic):
                sum_phi.append(sum(phi[k]))
            ndk = theta
            nkv = np.zeros((self.n_topic, n_word_types))
            
            sample_X = []
            for d in range(n_documents):
                n_word_in_doc = len(word_indexes[d])
                sum_theta_d = sum(theta[d])
                prob_d = digamma(theta[d]) - digamma(sum_theta_d)
                temp1 = svm_alpha[d] / n_word_in_doc * sample_y[d]
                
                ndk[d, :] = 0.
                dummies = np.array([0.] * self.n_topic)
                for w in range(n_word_in_doc):
                    temp2 = temp1 * reg_weights[k]
                    word_no = word_indexes[d][w]
                    prob_w = digamma(phi[:, word_no]) - digamma(sum_phi)
                    latent_z = np.exp(prob_w + prob_d + temp2)
                    latent_z /= np.sum(latent_z)
                    
                    ndk[d, :] += latent_z * word_counts[d][w]
                    nkv[:, word_no] += latent_z * word_counts[d][w]
                    
                    z = np.argmax(latent_z)
                    dummies[z] += 1.
                sample_X.append(dummies / len(word_indexes[d]))
                
            theta = ndk + self.alpha
            phi = nkv + self.beta
            print(n, np.max(theta - old_theta))
            old_theta = np.copy(theta)
            t1 = time.time()
            
            sample_X = np.array(sample_X)
            trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1)
            trainer.train(sample_X, sample_y.reshape(-1, 1))
            svm_alpha = trainer.lagrange_multipliers
            reg_weights = (svm_alpha * sample_y).T.dot(sample_X)
            self.b = (sample_y - sample_X.dot(reg_weights)).mean()
            print("svm train time:%.2fsec"%(time.time() - t1))
            self.reg_weights = reg_weights
            y_est = slda.predict(sample_X)
            print("current accuracy", accuracy_score(y_est, sample_y))
            
        for k in range(self.n_topic):
            phi[k] = phi[k] / np.sum(phi[k])

        for d in range(n_documents):
            theta[d] = theta[d] / np.sum(theta[d])
            
        self.reg_weights = reg_weights
        return phi, theta, sample_X
    

    def predict(self, sample_X):
        d_est = []
        for X in sample_X:
            mu = np.dot(self.reg_weights, X)
            d_est.append(1 if mu + self.b > 0 else -1)
            
        return d_est

if __name__ == "__main__":   
    import pandas as pd
    from gensim import corpora, models, similarities
    np.random.seed(12345)
    
    n_train = 5000
    n_topic = 200
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train = pd.read_csv('../data/train.csv').fillna(' ')
    
    train_text = train['comment_text']
    train_text0 = train_text[train['toxic']==0][:n_train / 2]
    train_text1 = train_text[train['toxic']==1][:n_train / 2]
    train_text = pd.concat([train_text0, train_text1]).as_matrix()
    train_target0 = train['toxic'][train['toxic']==0][:n_train / 2]
    train_target1 = train['toxic'][train['toxic']==1][:n_train / 2]
    train_target = pd.concat([train_target0, train_target1]).as_matrix()
    train_target[train_target==0] = -1
    train_target = train_target.astype(float)
    print(train_target)
    
    """
    train_target = np.array([-1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., 1.])
    train_text = ['This is body',
                    'This is me',
                    'I am good',
                    'I am teacher',
                    'This is grance',
                    'This is JAPAN',
                    'I am USA',
                     'I am China', 
                     'This is OK',
                     'This is miracle',
                     'I am place',
                     'I am grance',
                     'This is I am' # <- main problem
                 ]
    """
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
	             for document in train_text]
	             
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    slda = SLDA(n_topic, 100)
    phi, theta, sample_X = slda.fit(corpus, train_target)
    y_est = slda.predict(sample_X)
    print(y_est)
    print("accuracy", accuracy_score(y_est, train_target))
    #print(theta) 
     
    lda = vb.LDA(n_topic, 100)
    phi, theta, sampe_X = lda.fit(corpus)
    #svm = SVM(svm_alpha, n_iter=20000)
    #reg_weights, self.b, svm_alpha = svm.fit(sample_X, sample_y)
    
    trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1)
    predictor = trainer.train(sample_X, train_target.reshape(-1, 1))
    y_est = []
    for x in sample_X:
        y_est.append(predictor.predict(x))
    print("accuracy", accuracy_score(y_est, train_target))
    
    np.set_printoptions(precision=3, suppress=True)
    
