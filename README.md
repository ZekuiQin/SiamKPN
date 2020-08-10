# SiamKPN
<br/>Pytorch implementation of our paper [Siamese Keypoint Prediction Network for Visual Object Tracking](https://arxiv.org/abs/2006.04078).</br> 
Contact: [qinke87@gmail.com](qinke87@gmail.com)

## Intorduction
In this paper, we propose the Siamese keypoint prediction network (SiamKPN). Upon a Siamese backbone for feature embedding, SiamKPN benefits from a cascade heatmap strategy for coarseto-fine prediction modeling.
![Framework of SiamKPN](https://github.com/ZekuiQin/SiamKPN/blob/master/images/framework.png)

## Main Results

### Results on OTB-100
|   Traker  | AUC | Pre | Var decay| Speed | Model |
|:---------:|:---:|:---:|:--------:|:-----:|:-----:|
|SiamKPN-1s |0.687|0.906|    Yes   | 40FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|
|SiamKPN-2s |0.702|0.916|    Yes   | 32FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|
|SiamKPN-3s |0.712|0.927|    Yes   | 24FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|
|SiamKPN-3s |0.705|0.916|     No   | 24FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|    
### Results on VOT2018
|   Traker  | EAO |   A |  R  | Var decay| Speed | Model |
|:---------:|:---:|:---:|:---:|:--------:|:-----:|:-----:|
|SiamKPN-1s |0.413|0.584|0.229|   Yes    | 40FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|
|SiamKPN-2s |0.428|0.595|0.211|   Yes    | 32FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|
|SiamKPN-3s |0.440|0.605|0.187|   Yes    | 24FPS |[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw)|

Note:
- Speed tested on GTX-1080Ti.
- SiamKPN-1s refers to SiamKPN with one stage and so on.
- SiamKPN models on OTB-100 were trained with 20% random gray splits.
- We use modified ResNet-50 as the backbone model[link](https://pan.baidu.com/s/1MJwiYxXifKk5h43hmHYVpw).
- Models Extraction Code: gw6s

## Usage
Please find installation instructions in [INSTALL.md](https://github.com/ZekuiQin/SiamKPN/blob/master/INSTALL.md).
### Add SiamKPN to your PYTHONPATH
```export PYTHONPATH=/path/to/SiamKPN:$PYTHONPATH```
### Train
- Please prepare training datasets and testing datasets refer to [pysot](https://github.com/STVIR/pysot#introduction).
- Change the dataset paths to yours in pysot/datasets/dataset.py, tools/test.py and tools/eval.py.  

Take the usage of SiamKPN-3s_VOT as an example.
```
cd experiments/siamkpn_r50_stack3_difstd
CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    ../../tools/train.py --cfg config.yaml > logs/train.log
```
### Test
```
python ../../tools/test.py \
    --snapshot siamkpn-3s_vot.pth \
    --config config.yaml \
    --dataset VOT2018 
```
### Eval
```
python ../../tools/eval.py   \
    --tracker_path ./results \
    --dataset VOT2018
```
### Hyper-parameters Search
A two-level grid search is employed to find the best configuration.
```
for k in 0 1 2 3; do {
    for i in 0 1; do {
        echo "GPU $k, task $i"
        CUDA_VISIBLE_DEVICES=$k python -u ../../tools/hp_search_epoch.py \
                --start-epoch 11 \
                --end-epoch 12 \
        --penalty-k 0.0,0.5,0.1 \
        --window-influence 0.3,0.8,0.1 \
        --lr 0.3,0.8,0.1 \
        --config config.yaml \
        --dataset VOT2018
    } &
    done
    wait $!
} &
done
wait $!
```
## Acknowledgement
Our code is based on [pysot](https://github.com/STVIR/pysot#introduction).

We claim SiamKPN is the first to consider the anchor-free scheme in the Siamese paradigm for object tracking as we submitted our paper to CVPR 2020 in November 2019, though unfortunately get rejected. Concurrently, there were four other trackers considering the anchor-free scheme.
- FCAF([paper](https://ieeexplore.ieee.org/abstract/document/8817955))
- SiamFC++([paper](https://arxiv.org/abs/1911.06188))
- SiamBAN([paper](https://arxiv.org/abs/2003.06761)|[code](https://github.com/hqucv/siamban))
- FCOT([paper](https://arxiv.org/abs/2004.07109)|[code](https://github.com/MCG-NJU/FCOT))

Apart from the same motivation, SiamKPN has its own characteristics.
In particular, SiamKPN considered the cascade strategy for the anchor-free scheme to handle background distractors during tracking.

## Citation

	@article{Li_2020_SiamKPN,  
  	  title={Siamese Keypoint Prediction Network for Visual Object Tracking},  
  	  author={Li, Qiang and Qin, Zekui and Zhang, Wenbo and Zheng, Wen},  
   	  journal={arXiv preprint arXiv:2006.04078},  
  	  year={2020}  
	}