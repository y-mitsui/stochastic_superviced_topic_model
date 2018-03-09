
from __future__ import print_function
import subprocess
import numpy as np
import sys
import uuid
import os

class SLDA:
    def __init__(self, n_topics=10, alpha=1e-1, var_max_iter=20, var_convergence=1e-3, em_convergence=1e-4, em_max_iter=50, l2_penalty=1e-4):
        self.n_topics = n_topics
        self.alpha = alpha
        self.var_max_iter = var_max_iter
        self.var_convergence = var_convergence
        self.em_max_iter = em_max_iter
        self.em_convergence = em_convergence
        self.l2_penalty = l2_penalty
        self.cmd_dir = "/Users/mitsuiyosuke/Documents/workspace/slda/"
    
    def fit(self, corpus, label):
        self.temp_dir = "/tmp/slda_%s"%(uuid.uuid4())
        os.mkdir(self.temp_dir)
        
        with open(self.temp_dir + "/train_corpus.dat", "w") as fh:
            for row in corpus:
                text = " ".join(["%d:%d"%(w_i, w_c) for w_i, w_c in row])
                fh.write("%d %s\n"%(len(row), text))
                
        with open(self.temp_dir + "/train_label.dat", "w") as fh:
            fh.write("\n".join(map(str, label)) + "\n")
        
        with open(self.temp_dir + "/settings.txt", "w") as fh:
            fh.write("var max iter %d\n"%(self.var_max_iter))
            fh.write("var convergence %.5f\n"%(self.var_convergence))
            fh.write("em max iter %d\n"%(self.em_max_iter))
            fh.write("em convergence %.5f\n"%(self.em_convergence))
            fh.write("L2 penalty %.5f\n"%(self.l2_penalty))
            fh.write("alpha estimate\n")
            
        cmd = [
                self.cmd_dir + 'slda',
                'est',
                self.temp_dir + '/train_corpus.dat',
                self.temp_dir + '/train_label.dat',
                self.temp_dir + "/settings.txt",
                '%.5f'%(self.alpha),
                '%d'%(self.n_topics),
                'seeded',
                self.temp_dir,
           ]
        print(" ".join(cmd))
        res = subprocess.check_call(cmd)
        
    def predict(self, corpus, label, score=False):
        with open(self.temp_dir + "/test_corpus.dat", "w") as fh:
            for row in corpus:
                text = " ".join(["%d:%d"%(w_i, w_c) for w_i, w_c in row])
                fh.write("%d %s\n"%(len(row), text))
                
        with open(self.temp_dir + "/test_label.dat", "w") as fh:
            fh.write("\n".join(map(str, label)) + "\n")
            
        cmd = [
                self.cmd_dir + 'slda',
                'inf',
                self.temp_dir + '/test_corpus.dat',
                self.temp_dir + '/test_label.dat',
                self.temp_dir + "/settings.txt",
                self.temp_dir + "/final.model",
                self.temp_dir,
           ]
        print(" ".join(cmd))
        res = subprocess.check_call(cmd)
        if score:
            return np.loadtxt(self.temp_dir + '/inf-score.dat')
        else:
            return np.array(map(int, file(self.temp_dir + '/inf-labels.dat', 'r').readlines()))
    
    def predict_proba(self, corpus, label):
        return self.predict(corpus, label, True)
    
    
    def __del__(self):
        pass
        #res = subprocess.check_call(["rm", "-rf", self.temp_dir])
        
if __name__ == "__main__":
    
    corpus = [
                [(0, 1), (2, 1)],
                [(0, 1), (2, 1)],
                [(0, 1), (2, 1)],
                [(1, 1)],
                [(1, 1)],
                [(1, 1)],
                [(0, 1), (2, 2)],
                [(0, 1), (2, 2)],
                [(0, 1), (2, 2)],
          ]
    label = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    slda = SLDA(3, alpha=1., l2_penalty=1e-5)
    slda.fit(corpus, label)
    print(slda.predict(corpus, label))
    
    
