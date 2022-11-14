# PDP-N2S
N2S is a learning based improvement framework for solving the pickup and delivery problems, a.k.a. PDPs (e.g., PDTSP and PDTSP-LIFO).
It explores 3 advances on the top of [DACT](https://github.com/yining043/VRP-DACT):
- Synthesis Attention (Synth-Att), which achieves slightly better performance while largely reducing computational costs of DAC-Att in DACT
- two customized decoders, which learns to perform removal and reinsertion of a pickup-delivery node pair to tackle the precedence constraint
- a diversity enhancement scheme, which further ameliorates the performance

![](pdp.gif)

# Paper
![architecture](framework.jpg)

This repo implements our paper:

Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Hongliang Guo, Yuejiao Gong and Yeow Meng Chee, “[Efficient Neural Neighborhood Search for Pickup and Delivery Problems](https://arxiv.org/abs/2204.11399),” in the 31st International Joint Conference on Artificial Intelligence (IJCAI), 2022.

Please cite our paper if the code is useful for your project.
```
@inproceedings{ma2022efficient,
  title     = {Efficient Neural Neighborhood Search for Pickup and Delivery Problems},
  author    = {Ma, Yining and Li, Jingwen and Cao, Zhiguang and Song, Wen and Guo, Hongliang and Gong, Yuejiao and Chee, Yeow Meng},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4776--4784},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/662},
  url       = {https://doi.org/10.24963/ijcai.2022/662},
}

```
You may also find our [DACT](https://github.com/yining043/VRP-DACT) useful.
```
@inproceedings{ma2021learning,
 author = {Ma, Yining and Li, Jingwen and Cao, Zhiguang and Song, Wen and Zhang, Le and Chen, Zhenghua and Tang, Jing},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {11096--11107},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer},
 url = {https://proceedings.neurips.cc/paper/2021/file/5c53292c032b6cb8510041c54274e65f-Paper.pdf},
 volume = {34},
 year = {2021}
}

```

# Jupyter Notebook
Note that following the data structure of our [DACT](https://github.com/yining043/VRP-DACT), we use linked list to store solutions. We thus highly recommend you to read our Jupyter notebook for DACT before getting into details of our code for N2S. Please open the [Jupyter notebook](https://github.com/yining043/VRP-DACT/blob/main/Play_with_DACT.ipynb) here :)


# Dependencies
* Python>=3.6
* PyTorch>=1.1
* numpy
* tensorboard_logger
* tqdm

# Usage
## Generating data
Training data is generated on the fly. Please follow repo [Demon0312/Heterogeneous-Attentions-PDP-DRL](https://github.com/Demon0312/Heterogeneous-Attentions-PDP-DRL) to generate validating or test data if needed. We also provide some randomly generated data in the  [datasets](./datasets) folder.

## Training
### PDTSP examples
20 nodes:
```python
CUDA_VISIBLE_DEVICES=0 python run.py --problem pdtsp --graph_size 20 --warm_up 2 --max_grad_norm 0.05 --val_m 1 --val_dataset './datasets/pdp_20.pkl' --run_name 'example_training_PDTSP20'
```

50 nodes:
```python
CUDA_VISIBLE_DEVICES=0,1 python run.py --problem pdtsp --graph_size 50 --warm_up 1.5 --max_grad_norm 0.15 --val_m 1 --val_dataset './datasets/pdp_50.pkl' --run_name 'example_training_PDTSP50'
```

100 nodes:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --problem pdtsp --graph_size 100 --warm_up 1 --max_grad_norm 0.35 --val_m 1 --val_dataset './datasets/pdp_100.pkl' --run_name 'example_training_PDTSP100'
```
### PDTSP-LIFO examples
20 nodes:
```python
CUDA_VISIBLE_DEVICES=0 python run.py --problem pdtspl --graph_size 20 --warm_up 2 --max_grad_norm 0.05 --val_m 1 --val_dataset './datasets/pdp_20.pkl' --run_name 'example_training_PDTSPL20'
```

50 nodes:
```python
CUDA_VISIBLE_DEVICES=0,1 python run.py --problem pdtspl --graph_size 50 --warm_up 1.5 --max_grad_norm 0.15 --val_m 1 --val_dataset './datasets/pdp_50.pkl' --run_name 'example_training_PDTSPL50'
```

100 nodes:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --problem pdtspl --graph_size 100 --warm_up 1 --max_grad_norm 0.35 --val_m 1 --val_dataset './datasets/pdp_100.pkl' --run_name 'example_training_PDTSPL100'
```

### Warm start
You can initialize a run using a pretrained model by adding the --load_path option:
```python
--load_path '{add model to load here}'
```
### Resume Training
You can resume a training by adding the --resume option:
```python
--resume '{add last saved checkpoint(model) to resume here}'
```
The Tensorboard logs will be saved to folder "logs" and the trained model (checkpoint) will be saved to folder "outputs". Pretrained models are provided in the [pre-trained](./pre-trained) folders.

## Inference
Load the model and specify the iteration T for inference (using --val_m for data augments):

```python
--eval_only 
--load_path '{add model to load here}'
--T_max 3000 
--val_size 2000 
--val_batch_size 200 
--val_dataset '{add dataset here}' 
--val_m 50
```

### Examples
For inference 2,000 PDTSP instances with 100 nodes and no data augment (N2S):
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --eval_only --no_saving --no_tb --problem pdtsp --graph_size 100 --val_m 1 --val_dataset './datasets/pdp_100.pkl' --load_path 'pre-trained/pdtsp/100/epoch-195.pt' --val_size 2000 --val_batch_size 2000 --T_max 3000
```
For inference 2,000 PDTSP instances with 100 nodes using the augments in Algorithm 2 (N2S-A):
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --eval_only --no_saving --no_tb --problem pdtsp --graph_size 100 --val_m 50 --val_dataset './datasets/pdp_100.pkl' --load_path 'pre-trained/pdtsp/100/epoch-195.pt' --val_size 2000 --val_batch_size 200 --T_max 3000
```
See [options.py](./options.py) for detailed help on the meaning of each argument.

# Acknowledgements
The code and the framework are based on the repos [yining043/VRP-DACT](https://github.com/yining043/VRP-DACT).
