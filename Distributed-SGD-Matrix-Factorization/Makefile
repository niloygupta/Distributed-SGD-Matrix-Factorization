all:
	/Users/niloygupta/spark-1.3.0/bin/spark-submit aws_dsgd_mf.py 100 10 50 0.8 1.0 ./training_set w.csv h.csv
test:
	python eval_acc.py log /Users/niloygupta/spark-1.3.0/bin/spark-submit dsgd_mf.py 4 4 10 0.8 1.0 input.csv W.csv H.csv
	
eval:
	python eval_acc.py log /Users/niloygupta/spark-1.3.0/bin/spark-submit dsgd_mf.py 4 4 10 0.8 1.0 autolab_train.csv W.csv H.csv
	
acc:
	python eval_acc.py log /Users/niloygupta/spark-1.3.0/bin/spark-submit dsgd_mf.py 4 4 10 0.8 1.0 autolab_train.csv W.csv H.csv
	
big:
	/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin/
	

	
aws:
	./spark/bin/spark-submit --master spark://172.31.56.217:7077 aws_dsgd_mf.py 100 4 10 0.1 1.0 s3n://cmu-bigml/netflix-ratings/training_set/  w.csv h.csv
	
ashu:
	python eval_acc.py log /Users/niloygupta/spark-1.3.0/bin/spark-submit dsgd_mf.py 20 4 10 0.9 1.0 autolab_train.csv W.csv H.csv
	
	
optimized:
	 /Users/niloygupta/spark-1.3.0/bin/spark-submit testing.py 20 3 10 0.9 0.5 autolab_train.csv w.csv h.csv