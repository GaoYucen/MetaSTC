Environment:
Python 3.8
Pytorch 2.1.2

data:
- traffic_flow: samples for traffic flow data
- link_feature.txt: spatial features for roads

model_code:
- meta-LSTM: MAML+LSTM
- meta-film: MAML+LSTM
- Time-Series-Library: baselines
- ablation study: including clustering, distance function

To execute:
'python model_code/meta-LSTM.py'
'python model_code/meta-film.py'

