# Task-oriented-imputation
This repo is the Pytorch implementation of our submitted paper to NeurIPS 2024: Task-oriented Time Series Imputation Evaluation via Generalized Representers.

# Apply time series imputation
Users can directly run imputation.py to get the imputation result. To get the result shown in the paper, users need to change two parameters to change the data used like this.
~~~
python imputation.py -p './data/GEF.csv' -n 'GEF'
~~~

Note that the name must be in ['GEF','ETTH1','ETTH2','ELE','Traffic','Air'].

# Apply time series imputation ensemble
Users can run main.py to get the result shown in the paper. Note that the conde can not run without the corresponding train data (run the imputation.py to get them), and the result will be in the corresponding './result' folder.

~~~
python main.py
~~~
