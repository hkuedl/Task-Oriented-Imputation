# Task-oriented-imputation
This repo is the Pytorch implementation of our accepted paper to NeurIPS 2024: "Task-oriented Time Series Imputation Evaluation via Generalized Representers."

Authors: Zhixian Wang, Linxiao Yang, Liang Sun, Qingsong Wen, and Yi Wang.

## Apply time series imputation
Users can directly run imputation.py to get the imputation result. To get the result shown in the paper, users need to change two parameters to change the data used like this.
~~~
python imputation.py -p './data/GEF.csv' -n 'GEF'
~~~

Note that the name must be in ['GEF','ETTH1','ETTH2','ELE','Traffic','Air'].

## Apply time series imputation ensemble
Users can run main.py to get the result shown in the paper. Note that the code can not run without the corresponding train data (run the imputation.py to get them), and the result will be in the corresponding './result' folder.

~~~
python main.py
~~~

## Apply acceleration method

To use the acceleration method mentioned in the paper, users need to replace the 
```python
phi_trace,model = compute_phi_trace(model, torch.Tensor(train_data), torch.Tensor(train_label), torch.Tensor(test_data), torch.Tensor(test_label), learning_rate, epochs,train_criterion = nn.MSELoss(),device=device)
```
with
```python
phi_trace,model = compute_phi_trace_fast(model, torch.Tensor(train_data), torch.Tensor(train_label), torch.Tensor(test_data), torch.Tensor(test_label), learning_rate, epochs,train_criterion = nn.MSELoss(),device=device,num_segments = num_segments)
```
Users need to define num_segments to decide how many segments to divide the time series (default setting to 4).
