# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
from sklearn.metrics import accuracy_score
import sys
import time

class SLDA:

    def __init__(self, n_topic=10, n_iter=20):
        self.n_topic = n_topic
        self.n_iter = n_iter
    
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
            
        alpha = 1.
        beta = 1.
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
        reg_weights = np.random.normal(size=self.n_topic)
        slack = np.random.uniform(size=n_documents) + 1e-3
        
        latent_z = np.random.uniform(size=())
        latent_z = []
        for i in range(n_documents):
            temp = np.random.uniform(size=(len(word_indexes[i]), self.n_topic)) + 1e-5
            temp /= np.sum(temp, 1).reshape(-1, 1)
            latent_z.append(temp)
        
        n_word_in_docs = []
        for d in range(n_documents):
            n_word_in_docs.append(len(word_indexes[d]))
                
        for n in range(self.n_iter):
            sum_phi = []
            for k in range(self.n_topic):
                sum_phi.append(sum(phi[k]))
            
            for d in range(n_documents):
                n_word_in_doc = len(word_indexes[d])
                sum_theta_d = sum(theta[d] + alpha)
                diga_theta = digamma(theta[d] + alpha) - sum_theta_d
                
                z_reg_w_tot = 1.
                for w in range(n_word_in_doc):
                    temp = latent_z[d][w] * np.exp(reg_weights / n_word_in_doc)
                    z_reg_w_tot *= temp.sum()
                
                for w in range(n_word_in_doc):
                    temp = latent_z[d][w] * np.exp(reg_weights / n_word_in_doc)
                    z_reg_w = z_reg_w_tot / temp.sum()
                    
                    temp3 = sample_y[d] * reg_weights / n_word_in_doc
                    temp4 = (1. / slack[d]) * np.exp(reg_weights / n_word_in_doc) * z_reg_w
                    temp5 = temp3 - temp4
                
                    word_no = word_indexes[d][w]
                    k_sum = 0.
                    for k in range(self.n_topic):
                        prob_w = digamma(phi[k][word_no] + beta) - digamma(sum_phi[k] + beta)
                        prob_d = diga_theta[k]
                        latent_z[d][w][k] = np.exp(prob_w + prob_d + temp5[k])
                        k_sum += latent_z[d][w][k]
                    latent_z[d][w] /= k_sum
                
                for k in range(self.n_topic):
                    theta[d, k] = (latent_z[d][:, k] * word_counts[d]).sum()
            
                temp2 = 1.
                for w in range(n_word_in_doc):
                    temp = latent_z[d][w] * (1 + np.exp(reg_weights / n_word_in_doc))
                    temp2 *= temp.sum()
                slack[d] = temp2
                    
            for k in range(self.n_topic):
                for v in range(n_word_types):
                    tmp = 0.
                    for d in range(n_documents):
                        index = np.array(word_indexes[d]) == v
                        target_word_counts = np.array(word_counts[d])[index]
                        if target_word_counts.shape[0] != 0:
                            tmp += latent_z[d][index, k] * target_word_counts
                    phi[k][v] = tmp
            
            temp3s = []
            for d in range(n_documents):
                row_temp3 = []
                for w in range(n_word_in_docs[d]):
                    row_temp3.append(latent_z[d][w] / n_word_in_docs[d])
                temp3s.append(row_temp3)
                
            def deltaRegW(reg_weights):
                delta = np.zeros(len(reg_weights))
                for d in range(n_documents):
                    z_reg_w_tot = 1.
                    temp_exp = np.exp(reg_weights / n_word_in_docs[d])
                    
                    for w in range(n_word_in_docs[d]):
                        temp = latent_z[d][w] * temp_exp
                        z_reg_w_tot *= temp.sum()
                    
                    temp1 = sample_y[d] * np.average(latent_z[d], 0)
                    temp2 = 0.
                    for w in range(n_word_in_docs[d]):
                        temp = latent_z[d][w] * temp_exp
                        z_reg_w = z_reg_w_tot / temp.sum()
                        temp2 += temp3s[d][w] * temp_exp * z_reg_w
                    delta += temp1 - temp2
                if np.random.uniform() < 0.01:
                    print(np.max(delta))
                return delta
            
            t1 = time.time()
            for i in range(2000):
                reg_weights += 0.0002 * deltaRegW(reg_weights)
            print("time", time.time() - t1)
            #print(reg_weights)
            
            self.reg_weights = reg_weights
            self.latent_z = latent_z
            y_est = slda.predict()
            print("y_est", y_est)
            print("accuracy0", accuracy_score(y_est, sample_y))
            
            print("convergence theta",n, np.max(theta - old_theta))
            old_theta = np.copy(theta)
        
        for k in range(self.n_topic):
            phi[k] = phi[k] / np.sum(phi[k])

        for d in range(n_documents):
            theta[d] = theta[d] / np.sum(theta[d])
            
        self.reg_weights = reg_weights
        self.latent_z = latent_z
        return phi, theta

    def predict(self):
        d_est = []
        for d in range(len(self.latent_z)):
            z = np.argmax(self.latent_z[d], 1)
            dummies = []
            for each_z in z:
                val = [0] * self.n_topic
                val[each_z] = 1
                dummies.append(val)
            mu = np.dot(self.reg_weights, np.average(dummies, 0))
            a = np.exp(mu)
            d_est.append(a / (1 + np.exp(mu)))
            
        return np.array([int(x > 0.5) for x in d_est])

if __name__ == "__main__":   
    import pandas as pd
    from gensim import corpora, models, similarities
         
    np.random.seed(12345)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train = pd.read_csv('../data/train.csv').fillna(' ')

    n_train = 150

    train_text = train['comment_text']
    train_text0 = train_text[train['toxic']==0][:n_train / 2]
    train_text1 = train_text[train['toxic']==1][:n_train / 2]
    train_text = pd.concat([train_text0, train_text1]).as_matrix()
    train_target0 = train['toxic'][train['toxic']==0][:n_train / 2]
    train_target1 = train['toxic'][train['toxic']==1][:n_train / 2]
    train_target = pd.concat([train_target0, train_target1]).as_matrix()
    print(train_target)
    
    #train_target = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    
    """
    train_text = ['This is mother',
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
                     'I am grance']
    """
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
	             for document in train_text]
	             
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    slda = SLDA(100, 30)
    print(corpus)
    phi, theta = slda.fit(corpus, train_target)
    y_est = slda.predict()
    print(y_est)
    print("accuracy", accuracy_score(y_est, train_target))
    
    np.set_printoptions(precision=3, suppress=True)
    print(theta)  
