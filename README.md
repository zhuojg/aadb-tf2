# AADB TensorFlow
Convert [AADB](https://github.com/aimerykong/deepImageAestheticsAnalysis) caffe model to TensorFlow. And run it with a [Streamlit](https://www.streamlit.io/) Demo.

## Run Demo
* Environment Configuration
```shell script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir data
```
* Download [`initModel.data`](https://drive.google.com/open?id=1A_daf5zCU68fU1HZRQcwhxbq5MK5z1OD) and [`mean_AADB.txt`](https://drive.google.com/open?id=1asZNyLoQGTtJAINqyak7KmtwZfWH3YCC) and move them to `data` folder.
* Run Demo
```shell script
streamlit run demo.py
```

## Attention and TODO
Although the code can run with TensorFlow >= 2.0.0, the whole structure is actually based on **TensorFlow 1.x**.  
So I need to convert make code **2.0-native**.

## Convert Method
Use repo [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) to convert original caffe model to TensorFlow code and model. 
The repo is really old, it only support python2 and TensorFlow < 1.0.0.   
I convert the model successfully with following environment on `Ubuntu 18.04`:  
```shell script
python==2.7
tensorflow==0.12.0
numpy==1.16.1
```  
Besides, I build caffe with this [guide](https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215).
