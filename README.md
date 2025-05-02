# (NeurIPS 2024) Task-oriented Time Series Imputation Evaluation via Generalized Representers
This repo is the Pytorch implementation of our NeurIPS'24 paper:
- Zhixian Wang, Linxiao Yang, Liang Sun, Qingsong Wen*, Yi Wang*, "Task-oriented Time Series Imputation Evaluation via Generalized Representers," in 38th Annual Conference on Neural Information Processing Systems (NeurIPS 2024), Vancouver, Canada, Dec. 2024.

## Citation
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```tex
@article{wang2024task,
  title={Task-oriented Time Series Imputation Evaluation via Generalized Representers},
  author={Wang, Zhixian and Yang, Linxiao and Sun, Liang and Wen, Qingsong and Wang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={137403--137431},
  year={2024}
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.



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

## Acknowledgement

We appreciate the following [\[PyPOTS repo\]](https://github.com/WenjieDu/PyPOTS) for providing the time series imputation code.

## Further Reading
1, Deep Learning for Multivariate Time Series Imputation: A Survey, in *arXiv* 2024. 
[\[paper\]](https://arxiv.org/abs/2402.04059) [\[Website\]](https://github.com/wenjiedu/awesome_imputation)

2, TSI-Bench: Benchmarking Time Series Imputation, in *arXiv* 2024. 
[\[paper\]](https://arxiv.org/abs/2406.12747) [\[Website\]](https://github.com/wenjiedu/awesome_imputation)

3, AI for Time Series (AI4TS) Papers, Tutorials, and Surveys. 
[\[Website\]](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)
