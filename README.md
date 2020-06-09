# SiamKPN
Pytorch implementation of our paper [Siamese Keypoint Prediction Network for Visual Object Tracking](https://arxiv.org/abs/2006.04078).
## Intorduction
In this paper, we propose the Siamese keypoint prediction network (SiamKPN). Upon a Siamese backbone for feature embedding, SiamKPN benefits from a cascade heatmap strategy for coarseto-fine prediction modeling.
![Framework of SiamKPN](https://github.com/ZekuiQin/SiamKPN/blob/master/images/framework.pdf)
## Main Results
### Results on OTB-100
Our SiamKPN-3s achieves a success score of 0.712 and precision score of 0.927 on OTB-100.     
### Results on VOT2018
Our SiamKPN-3s achieves an EAO score of 0.440 on VOT2018.  
## Acknowledgement
Our code is based on [pysot](https://github.com/STVIR/pysot#introduction).
## Citation

	@inproceedings{Li_2020_SiamKPN,  
  	  title={Siamese Keypoint Prediction Network for Visual Object Tracking},  
  	  author={Li, Qin and Zhang, Zheng},  
   	  booktitle={arXiv preprint arXiv:2006.04078},  
  	  year={2020}  
	}