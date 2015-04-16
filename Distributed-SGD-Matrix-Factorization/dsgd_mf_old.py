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
import random
import csv
beta_value = 0.0
lambda_value = 0.0

class blockType:
    def __init__(self, Vmin, Hmin,Wmin,blockRowIndex,blockColIndex):
        self.Vmin = Vmin
        self.Hmin = Hmin
        self.Wmin = Wmin
        self.blockRowIndex = blockRowIndex
        self.blockColIndex = blockColIndex


def load_data(inputV_filepath,num_factors,num_workers,sc):
    files = sc.wholeTextFiles(inputV_filepath).collect()
    data = lil_matrix((17770,2649429),dtype=np.float64)
    
    #for key in files:
    for fnum in range(0,len(files)):
        value = files[fnum][1]
        lines = value.split('\n')
        movie_id = int(lines[0].split(':')[0]) -1
        for i in range(1,len(lines)-1):
            rating_data = lines[i]
            user_ratings = rating_data.split(",")
            data[movie_id,int(user_ratings[0])-1] = float(user_ratings[1])
        files[fnum]=(files[fnum][0],'')
    data = csr_matrix(data)
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))
    return data,W,H,block_size


def updateGradient(block):
    
    V = block[0]
    H = block[1]
    W = block[2]
    mbi = block[5]
    
    rows,cols = V.nonzero()
    count = 0
    oldLoss = 0
    newLoss = 1
    #for i in range(0,100):
    while math.fabs(newLoss -oldLoss )>1e-5 and len(rows>0):
        oldLoss = newLoss
        k = random.randint(0,(rows).size-1)
        r = rows[k]
        c = cols[k]
        lr = math.pow((100 + 100*mbi+count),-beta_value)
        #lr = math.pow((100 + mbi),-beta_value)
        count = count + 1
        V_val = V[r,c]
        if V_val == -1:
            V_val = 0.0;
        
        Wgrad = -2*(V_val - W[r,:].dot(H[:,c]))*(H[:,c].T) + (2*lambda_value*W[r,:])/(V[r,:].nonzero()[0].size)
        Hgrad = -2*(V_val - W[r,:].dot(H[:,c]))*(W[r,:].T) + (2*lambda_value*H[:,c])/(V[:,c].nonzero()[0].size)
        H[:,c] = H[:,c] - lr*Hgrad
        W[r,:] = W[r,:] - lr*Wgrad
        
        V[V==-1] =  0
        P = W.dot(H)
        newLoss = l2Loss(V[rows,cols],P[rows,cols],W,H,lambda_value)

    return [W,H,block[3],block[4],mbi+count]
    
def l2Loss(V,P,W,H,lambda_value):
    l = V- P
    l = np.multiply(l,l) 
    return np.sum(l) +lambda_value*(np.sum(np.multiply(W,W)) + np.sum(np.multiply(H,H)))
       

def factorize_matrix(data,W,H,block_size,T,num_workers,sc):
    adjRow = data.shape[0]%num_workers
    adjCol = data.shape[1]%num_workers
    for epoch in range(0,T):
        mbi = 0
        for stratum in range(0,num_workers):
            blocks = []
            for b in range(0,num_workers):
                blockRowIndex = np.array([b*block_size[0],(b+1)*block_size[0]])
                blockColIndex = np.array([b*block_size[1],(b+1)*block_size[1]]) 
    
                if b!=0:
                    blockRowIndex[1] = blockRowIndex[1] + adjRow
                    blockColIndex[1] = blockColIndex[1] + adjCol
                
                blockColIndex[0] = (blockColIndex[0] + stratum*block_size[1])%data.shape[1]
                
                if (blockColIndex[1] + stratum*block_size[1])%data.shape[1] !=0:
                    blockColIndex[1] = (blockColIndex[1] + stratum*block_size[1])%data.shape[1]
                else:
                    blockColIndex[1] = data.shape[1]
                    
                
                Vmin = data[blockRowIndex[0]:blockRowIndex[1],blockColIndex[0]:blockColIndex[1]]
                Hmin = H[:,blockColIndex[0]:blockColIndex[1]]
                Wmin = W[blockRowIndex[0]:blockRowIndex[1],:]
                blocks.append((Vmin,Hmin,Wmin,blockRowIndex,blockColIndex,mbi))
            grad_res = sc.parallelize(blocks,num_workers).map(lambda x:updateGradient(x)).collect()
            for res in grad_res:
                blockRowIndex, blockCoIndex = res[2], res[3]
                mbi = mbi + res[4]
                W[blockRowIndex[0]:blockRowIndex[1], :] = res[0]
                H[:, blockCoIndex[0]:blockCoIndex[1]] = res[1]
    return (W,H) 


def printToFile(W,outputW_filepath):
    f = open(outputW_filepath, 'w')
    for i in range(0,W.shape[0]):
        row = ''
        for j in range(W.shape[1]):
            row = row + "," +  str(W[i,j]) 
        f.write(row[1:] + "\n")
    f.close()
    
    
'''def load_csvdata(inputV_filepath,num_factors,num_workers,sc):
    files = sc.wholeTextFiles(inputV_filepath).collect()
    val = []
    row = []
    col = []
    for key,value in files:
        f = open(key)
        reader = csv.reader(f)
        for line in reader:
                row.append( int(line[0])-1 )
                col.append( int(line[1])-1 )
                rating = (line[2])
                if rating == '0':
                    rating = '-1'
                val.append( float(rating) )
    data = csr_matrix( (val, (row, col)) )
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))
    #W = np.ones((data.shape[0],num_factors),dtype=np.float)
    #H = np.ones((num_factors,data.shape[1]),dtype=np.float)
    return data,W,H,block_size'''

def load_csvdata(inputV_filepath,num_factors,num_workers,sc):
    val = []
    row = []
    col = []
    select = []
    f = open(inputV_filepath)
    reader = csv.reader(f)
    for line in reader:
        row.append( int(line[0])-1 )
        col.append( int(line[1])-1 )
        rating = (line[2])
        if rating == '0':
            rating = '-1'
        val.append( float(rating) )
        select.append( (int(line[0])-1, int(line[1])-1) )
    data = csr_matrix( (val, (row, col)) )
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))
    return data,W,H,block_size,select

def CalculateError(V, W, H, select):
    V[V==-1] =  0
    diff = V-W.dot(H)
    error = 0
    for row, col in select:
            error += diff[row, col]*diff[row, col]
    return error/len(select)
    
def main():
    global beta_value
    global lambda_value
    sc = SparkContext("local", "SGD-Matrix")
    print sys.argv
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]
    #[data,W,H,block_size] = load_data(inputV_filepath,num_factors,num_workers,sc)
    [data,W,H,block_size,select] = load_csvdata(inputV_filepath,num_factors,num_workers,sc)
    W,H = factorize_matrix(data,W,H,block_size,num_iterations,num_workers,sc)
    printToFile(W,outputW_filepath)
    printToFile(H,outputH_filepath)
    printToFile(W.dot(H),"predicted_ratings.csv")
    error  = CalculateError(data, W, H, select)
    print "Recon Error: " + str(error)
    
if __name__ == '__main__':
    main()