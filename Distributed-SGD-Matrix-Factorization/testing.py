import os
import sys
from pyspark import SparkContext, SparkConf
import math 
import numpy as np
import random

def dotProduct(M):
	return np.sum(M * M)

def computeGradient(block):
	#block = block.collect()
	Vmini, Wmini, Hmini, globalIter, index = block[0], block[1], block[2], block[3], block[4]
	blockEntryRow = block[5]
	blockEntryCol = block[6]
	#Vmini = V.collect()
	total = len(Vmini) 

	iteration = 0
	prevLoss = -1.0
	nonzeroCount = len(Vmini)
	while True and nonzeroCount > 0:
		randomIndex = random.randint(0, total-1)
		
		r = Vmini[randomIndex][1][0] - blockEntryRow
		c = Vmini[randomIndex][1][1] - blockEntryCol

		W = Wmini[r, :]
		H = Hmini[:, c]
		learningRate = math.pow((100 + globalIter +  iteration), -betaValue)
		

		temp = 2.0 * (Vmini[randomIndex][1][2] - W.dot(H))

		#vrnonzeros = 1.0 * V.filter(lambda x: x[0] == r).count()
		#vcnonzeros = 1.0 * V.filter(lambda x: x[1] == c).count()
		
		vrnonzeros = 1.0 * len(map(lambda x: x[1][0] == r, Vmini))
		vcnonzeros = 1.0 * len(map(lambda x: x[1][1] == c, Vmini))
		
		W += (learningRate * temp * H.T) - (2.0 * learningRate * (lambdaValue/vrnonzeros) * W)
		H += (learningRate * temp * Wmini[r, :].T) - (2.0 * learningRate * (lambdaValue/vcnonzeros) * H)
		Wmini[r, :] = W
		Hmini[:, c] = H
		#loss = Vnonzero - Wmini.dot(Hmini)[rowZ, colZ]
		#loss = dotProduct(loss) + lambdaValue * (dotProduct(Wmini) + dotProduct(Hmini))
		#if np.fabs(prevLoss - loss) < 0.000001:
		#	break
		#else:
		#	prevLoss = loss
		iteration += 1
		if iteration == 1000:
			break
	
	return (index, globalIter + iteration, Wmini, Hmini)

def createStratum(x, numWorkers, userBlockSize,movieBlockSize):
	#7, 23

	userId, movieId = x[0], x[1]
	temp = (movieId/movieBlockSize) - (userId/userBlockSize)
	temp = temp%numWorkers

	return temp

def writeOutput(npArr, outputFile):
	np.savetxt(outputFile, npArr, delimiter=",")

def main():
	
	global betaValue, lambdaValue
	noOfFactors = int(sys.argv[1])		  # Number of factors
	noOfWorkers = int(sys.argv[2])		  # Number of workers
	noOfIterations = int(sys.argv[3])	   # Number of iterations
	betaValue = float(sys.argv[4])		  # Beta value (decay parameter)
	lambdaValue = float(sys.argv[5])		# Regularization parameter
	inputPath = sys.argv[6]				 # Path to input (it can be a directory or a file)
	outputWPath = sys.argv[7]			   # Output file for writing W factored matrix in csv format
	outputHPath = sys.argv[8]			   # Output file for writing H factored matrix in csv format
	
	sc = SparkContext(appName="testing")
	tuples = sc.textFile(inputPath).map(lambda x: x.split(",")).map(lambda x: (int(x[0])-1, int(x[1])-1, float(x[2])))
	#tuples = sc.textFile(inputPath).flatMap(lambda x: x.split(","))
	
	noOfUsers = tuples.map(lambda x: x[0]).max() + 1
	noOfMovies = tuples.map(lambda x: x[1]).max() + 1
	print noOfUsers,noOfMovies
	
	W = np.random.random_sample((noOfUsers, noOfFactors))
	H = np.random.random_sample((noOfFactors, noOfMovies))
	remainRow, remainCol = noOfUsers%noOfWorkers, noOfMovies%noOfWorkers
	if remainRow > 0:
		remainRow = 1
	if remainCol > 0:
		remainCol = 1
	blockSize = ((noOfUsers/noOfWorkers) + remainRow, (noOfMovies/noOfWorkers) + remainCol) # Size of each block in a stratum

	partitions = tuples.keyBy(lambda x: createStratum(x, noOfWorkers, blockSize[0],blockSize[1]))
	#print partitions.collect()
	#partitions = tuples.map(lambda x: createStratum(x, noOfWorkers, blockSize[0],blockSize[1]))
	#print partitions.filter(lambda x: x[0] == 0).collect()
	#exit(0)

	
	stratumIndices = {}
	for stratum in xrange(noOfWorkers):                     # Creating stratum
		blocks = [] 
		for worker in xrange(noOfWorkers):                  # Creating blocks in a stratum
			blockEntryRow, blockExitRow = worker * blockSize[0], (worker + 1) * blockSize[0]
			blockEntryCol, blockExitCol = (worker + stratum) * blockSize[1], (stratum + worker + 1) * blockSize[1]
			if blockEntryCol > noOfMovies:
				blockEntryCol = (blockEntryCol % noOfMovies) - 1
				blockExitCol = blockEntryCol + blockSize[1]
			if blockExitRow > noOfUsers:
				blockExitRow = noOfUsers
			if blockExitCol > noOfMovies:
				blockExitCol = noOfMovies
			blocks.append((blockEntryRow, blockExitRow, blockEntryCol, blockExitCol))
		stratumIndices[stratum] = blocks
	for iteration in xrange(noOfIterations):
		
		globalIter = 0
		for stratum in range(2,noOfWorkers):#partitions.keys().collect():
			blocks = partitions.filter(lambda x: x[0] == stratum)

			temp = stratumIndices[stratum]
			blockInfoList = []
			for index, block in enumerate(temp):
				blockEntryRow, blockExitRow, blockEntryCol, blockExitCol = block[0], block[1], block[2], block[3]
				Wmini = W[blockEntryRow:blockExitRow, :]
				Hmini = H[:, blockEntryCol:blockExitCol]
				blockInfo = blocks.filter(lambda x: x[1][0]/blockSize[1] == index)			

				
				#exit(0)
				#blockInfo = blockInfo.map(lambda x: (x, Wmini, Hmini, globalIter, index, blockEntryRow, blockEntryCol))
				#print blockInfo.collect()
				#exit(0)
				blockInfoList.append((blockInfo.collect(), Wmini, Hmini, globalIter, index, blockEntryRow, blockEntryCol))
				#blockInfoList.append(blockInfo)
			
			result = sc.parallelize(blockInfoList, noOfWorkers).map(lambda x: computeGradient(x)).collect()
			#result = blockInfoList.par.foreach(computeGradient()).collect()
			#result = blockInfoList.map(computeGradient()).collect()
			#result = result.map(lambda x: computeGradient(x)).collect()
			for r in result:            # Updating W, H from the partial results of each worker
				blockNum = r[0]
				block = temp[blockNum]
				blockEntryRow, blockExitRow, blockEntryCol, blockExitCol = block[0], block[1], block[2], block[3]
				globalIter += r[1]
				W[blockEntryRow:blockExitRow, :] = r[2]
				H[:, blockEntryCol:blockExitCol] = r[3]
				
	writeOutput(W, outputWPath)
	writeOutput(H, outputHPath)
	#writeOutput(W.dot(H), "predicted_ratings.csv")
	sc.stop()
	
if __name__ == "__main__":
	main()