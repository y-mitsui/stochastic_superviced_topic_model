from __future__ import print_function
import numpy as np

class SVM:
    def __init__(self, alpha_init=None, n_iter=1000):
        self.alpha_init = alpha_init
        self.n_iter = n_iter
        
    def fit(self, sample_X, sample_y):
        N, d = sample_X.shape
        if self.alpha_init is None:
            alpha = np.zeros(N)
        else:
            alpha = self.alpha_init
        
        np.set_printoptions(precision=3, suppress=True)
        beta = 1.0
        eta_al = 0.0005 # update ratio of alpha
        eta_be = 0.005 # update ratio of beta
    
        for _itr in range(self.n_iter):
            for i in range(N):
                delta = 1 - (sample_y[i] * sample_X[i]).dot(alpha * sample_y * sample_X.T).sum() - beta * sample_y[i] * alpha.dot(sample_y)
                alpha[i] += eta_al * delta
            for i in range(N):
                beta += eta_be * alpha.dot(sample_y) ** 2 / 2
            
            if (_itr + 1) % int(self.n_iter / 10) == 0:
                print("svm convargence:%.5f"%(np.max(delta)))
                print(alpha[:20])
                
        index = alpha > 0
        w = (alpha * sample_y).T.dot(sample_X)
        b = (sample_y[index] - sample_X[index].dot(w)).mean()
        return w, b, alpha


if __name__ == '__main__':
    def f(x, y):
        return x - y #+ np.random.normal()
    
    from matplotlib import pyplot
    import sys
    
    param = sys.argv

    np.random.seed(12345)
    N = 10
    d = 2
    X = np.random.randn(N, d)
    T = np.array([1 if f(x, y) > 0 else - 1 for x, y in X])
    svm = SVM(n_iter=10000)
    w, b, alpha = svm.fit(X, T)

    if '-d' in param or '-s' in param:
        seq = np.arange(-3, 3, 0.02)
        pyplot.figure(figsize = (6, 6))
        pyplot.xlim(-3, 3)
        pyplot.ylim(-3, 3)
        pyplot.plot(seq, -(w[0] * seq + b) / w[1], 'k-')
        pyplot.plot(X[T ==  1,0], X[T ==  1,1], 'ro', marker='o')
        pyplot.plot(X[T == -1,0], X[T == -1,1], 'bo', marker='^')

        if '-s' in param:
            pyplot.savefig('graph.png')

        if '-d' in param:
            pyplot.show()