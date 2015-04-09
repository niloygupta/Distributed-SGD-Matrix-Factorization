'''
Created on 09-Apr-2015

@author: niloygupta
'''

from pyspark import SparkContext

def main():
    sc = SparkContext("local", "SGD-Matrix")
    words = sc.textFile("/Users/niloygupta/spark-1.3.0/README.md")
    a = words.filter(lambda w: w.startswith("spar")).take(5)
    print a

if __name__ == '__main__':
    main()