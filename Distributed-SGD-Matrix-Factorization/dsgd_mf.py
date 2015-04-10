'''
Created on 09-Apr-2015

@author: niloygupta
'''

from pyspark import SparkContext
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np
import math
from random import randint
beta_value = 0.0
lambda_value = 0.0

class block:
    def __init__(self, Vmin, Hmin,Wmin):
        self.Vmin = Vmin
        self.Hmin = Hmin
        self.Wmin = Wmin


def load_data(inputV_filepath,num_factors,num_workers,sc):
    files = sc.wholeTextFiles(inputV_filepath).collect()
    data = lil_matrix((17770,2649429),dtype=np.float64)
    
    for key,value in files:
        lines = value.split('\n')
        movie_id = int(lines[0].split(':')[0])
        for i in range(1,len(lines)-1):
            rating_data = lines[i]
            user_ratings = rating_data.split(",")
            data[movie_id,int(user_ratings[0])] = float(user_ratings[1])
    data = csr_matrix(data)
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))
    return data,W,H,block_size

'''
for epoch t=1,....,T in sequence
–  for stratum s=1,...,K in sequence
•  for block b=1,... in stratum s in parallel
–  for triple (i,j,raFng) in block b in sequence
»  do SGD step for (i,j,raFng)
'''

def updateGradient(block):
    V = block.Vmin
    H = block.Hmin
    W = block.Wmin
    rows,cols = V.nonzero()
    for i in range(1,10):
        k = randint(len(rows))
        r = rows[k]
        c = cols[k]
        lr = math.pow((100 + i),-beta_value)
        Wgrad = -2*(V[r,c] - W[r,:].dot(H[:,c])*H[:,c].T) + (2*lambda_value*W[r,:])/(V[r,:].nonzero()[0].size)
        W[r,:] = W[r,:] - lr*Wgrad
        Hgrad = -2*(V[r,c] - W[r,:].dot(H[:,c])*W[r,:].T) + (2*lambda_value*H[:,c])/(V[:,c].nonzero()[0].size)
        H[:,c] = H[:,c] - lr*Hgrad
        

def factorize_matrix(data,W,H,block_size,T,num_workers,sc):
    for epoch in range(0,T):
        blocks = []
        adjRow = data.shape[0]%num_workers
        adjCol = data.shape[1]%num_workers
        for stratum in range(0,num_workers):
            for b in range(0,num_workers):
                if b==0:
                    blockRowIndex = (b*block_size[0],(stratum+1)*block_size[0])
                    blockColIndex = (b*block_size[1],(stratum+1)*block_size[1])
                else:
                    blockRowIndex = (b*block_size[0],(stratum+1)*block_size[0]) + adjRow
                    blockColIndex = (b*block_size[1],(stratum+1)*block_size[1]) + adjCol
                Vmin = data[blockRowIndex[0]:blockRowIndex[1],blockColIndex[0]:blockColIndex[1]]
                Hmin = H[:,blockColIndex[0]:blockColIndex[1]]
                Wmin = W[blockRowIndex[0]:blockRowIndex[1],:]
                blocks.append(block(Vmin,Hmin,Wmin))
        sc.parallelize(blocks).map(updateGradient).collect()

def main():
    global beta_value
    global lambda_value
    sc = SparkContext("local", "SGD-Matrix")
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]
    [data,W,H,block_size] = load_data(inputV_filepath,num_factors,num_workers,sc)
    factorize_matrix(data,W,H,block_size,num_iterations,num_workers,sc)
if __name__ == '__main__':
    main()