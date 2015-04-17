'''
Created on 09-Apr-2015

@author: niloygupta
'''

from pyspark import SparkContext,SparkConf
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np
import math
import random
import csv
beta_value = 0.0
lambda_value = 0.0
#fexp = open("expResults", 'w')

class blockType:
    def __init__(self, Vmin, Hmin,Wmin,blockRowIndex,blockColIndex):
        self.Vmin = Vmin
        self.Hmin = Hmin
        self.Wmin = Wmin
        self.blockRowIndex = blockRowIndex
        self.blockColIndex = blockColIndex


def mapLists(content):
    rows = []
    cols = []
    vals = []
    select = []
    
    rating_data = content.split("\n")
    movie_id = int(rating_data[0][:-1]) - 1
    lines = map(lambda x: x.split(","), rating_data[1:])

    for i in range(1,len(lines)-1):
        rows.append(int(lines[i][0]) - 1)
        cols.append(movie_id)
        rating = lines[i][1]
        if rating =='0':
            rating = '-1'
        vals.append(float(rating))
        select.append( (int(lines[i][0]) - 1, int(lines[i][1]) - 1) )
    
    return (vals, rows, cols,select) 

def mapListsCSV(content):
    rows = []
    cols = []
    vals = []
    select = []
    rating_data = content.split("\n")

    lines = map(lambda x: x.split(","), rating_data[1:])

    for i in range(1,len(lines)-1):
        rows.append(int(lines[i][0]) - 1)
        cols.append(int(lines[i][1]) - 1)
        rating = lines[i][2]
        if rating =='0':
            rating = '-1'
        vals.append(float(rating))
        select.append( (int(lines[i][0]) - 1, int(lines[i][1]) - 1) )
    
    return (vals, rows, cols,select) 

def reduceLists(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2],x[3] + y[3])



def load_data(inputV_filepath,num_factors,num_workers,sc):
    data = sc.wholeTextFiles(inputV_filepath).map(lambda x: mapListsCSV(x[1])).reduce(lambda x, y: reduceLists(x, y))
    select = data[3]
    data = csr_matrix((data[0], (data[1], data[2])))
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))

    return data,W,H,block_size,select


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
    while math.fabs(newLoss -oldLoss )>1e-5 and len(rows)>0:
        oldLoss = newLoss
        k = random.randint(0,(rows).size-1)
        r = rows[k]
        c = cols[k]
        lr = math.pow((100 + mbi+count),-beta_value)
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
       

def factorize_matrix(data,W,H,block_size,T,num_workers,sc,select):
    adjRow = data.shape[0]/num_workers
    adjCol = data.shape[1]/num_workers 
    if(data.shape[0]%num_workers !=0):
        adjRow = adjRow + 1
    if(data.shape[1]%num_workers !=0):
        adjCol = adjCol + 1

    #print adjRow,adjCol
    for epoch in range(0,T):
        mbi = 0
        for stratum in range(0,num_workers):
            blocks = []
            for b in range(0,num_workers):
                blockRowIndex = np.array([b*adjRow,(b+1)*adjRow])
                blockColIndex = np.array([(b+stratum)*adjCol,(b+stratum+1)*adjCol])
                
                if(blockColIndex[0] > data.shape[1]):
                    blockColIndex[0] = blockColIndex[0]%data.shape[1] - 1
                
                blockColIndex[1] = blockColIndex[0] + adjCol
                
                if(blockColIndex[1]>data.shape[1]):
                    blockColIndex[1] = data.shape[1]
                    
                if(blockRowIndex[1]>data.shape[0]):
                    blockRowIndex[1] = data.shape[0]
    
                
                
                
                #print blockRowIndex,blockColIndex
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
    
    
def load_csvdata(inputV_filepath,num_factors,num_workers,sc):
    files = sc.wholeTextFiles(inputV_filepath).collect()
    val = []
    row = []
    col = []
    select = []
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
            select.append( (int(line[0])-1, int(line[1])-1) )
    data = csr_matrix( (val, (row, col)) )
    block_size =  (data.shape[0]/num_workers,data.shape[1]/num_workers)
    W = np.random.random_sample((data.shape[0],num_factors))
    H = np.random.random_sample((num_factors,data.shape[1]))
    #W = np.ones((data.shape[0],num_factors),dtype=np.float)
    #H = np.ones((num_factors,data.shape[1]),dtype=np.float)
    return data,W,H,block_size,select


def CalculateError(V, W, H, select):
    V[V==-1] =  0
    diff = V-W.dot(H)
    error = 0
    for row, col in select:
            error += diff[row, col]*diff[row, col]
    error = error/len(select)
    print "\tRecon Error: " + str(error)+"\n"
    #fexp.write("\tRecon Error: " + str(error)+"\n")
    return error
    
def main():
    global beta_value
    global lambda_value
    #spark://23.195.26.187:7077
    #sc = SparkContext("local", "SGD-Matrix")
    #conf = SparkConf().setAppName("SGD-Matrix")#.setMaster("spark://172.31.56.217:7077")
    #sc = SparkContext("local[3]", "SGD-Matrix")
    #sc = SparkContext("local[3]", "SGD-Matrix")
    sc = SparkContext("spark://ip-172-31-58-176.ec2.internal:7077", "SGD-Matrix")
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]
    #[data,W,H,block_size,select] = load_data(inputV_filepath,num_factors,num_workers,sc)
    #[data,W,H,block_size,select] = load_csvdata(inputV_filepath,num_factors,num_workers,sc)
    beta =0.5
    while beta<=0.9:
        beta_value = beta
        [data,W,H,block_size,select] = load_data(inputV_filepath,num_factors,num_workers,sc)
        W,H = factorize_matrix(data,W,H,block_size,num_iterations,num_workers,sc,select)
    #np.savetxt(outputW_filepath,W,delimiter=',')
    #printToFile(W,outputW_filepath)
    #np.savetxt(outputH_filepath,H,delimiter=',')
    #printToFile(H,outputH_filepath)
    #printToFile(W.dot(H),"predicted_ratings.csv")
        CalculateError(data, W, H, select)
        beta=beta+0.1
    #fexp.close()
    
if __name__ == '__main__':
    main()