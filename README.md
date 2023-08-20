<div align="center">
<h1> RLIPv2: Fast Scaling of Relational Language-Image Pre-training
</h1>

<div>
    <a href='https://jacobyuan7.github.io/' target='_blank'>Hangjie Yuan</a>&emsp;
    <a href='https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ&hl=en&oi=ao' target='_blank'>Shiwei Zhang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=cQbXvkcAAAAJ&hl=en' target='_blank'>Xiang Wang</a>&emsp;
    <a href='https://samuelalbanie.com/' target='_blank'>Samuel Albanie</a>&emsp;
    <a href='https://pynsigrid.github.io/' target='_blank'>Yining Pan</a>&emsp;<br>
<!--     Yining Pan&emsp;<br> -->
    <a href='https://scholar.google.com/citations?user=JT8hRbgAAAAJ&hl=en' target='_blank'>Tao Feng</a>&emsp;
    <a href='https://scholar.google.com/citations?user=37gvStUAAAAJ&hl=en' target='_blank'>Jianwen Jiang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=boUZ-jwAAAAJ&hl=en' target='_blank'>Dong Ni&#9993</a>&emsp;
    <a href='https://scholar.google.com/citations?user=16RDSEUAAAAJ&hl=en' target='_blank'>Yingya Zhang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=7LhjCn0AAAAJ&hl=en' target='_blank'>Deli Zhao</a>&emsp;
</div>
    
<strong>Accepted to <a href='https://iccv2023.thecvf.com/' target='_blank'>ICCV 2023</a> :partying_face:</strong>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.01814)
[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/RLIPv2?style=social)](https://github.com/JacobYuan7/RLIPv2)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/RLIPv2)](https://github.com/JacobYuan7/RLIPv2)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJacobYuan7%2FRLIPv2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</div>

![colored_mesh (1)](assets/teaser.png)

> Abstract:
> Relational Language-Image Pre-training (RLIP) aims to align vision representations with relational texts, thereby advancing the capability of relational reasoning in computer vision tasks.
> However, hindered by the slow convergence of RLIPv1 architecture and the limited availability of existing scene graph data, scaling RLIPv1 is challenging.
> In this paper, we propose RLIPv2, a fast converging model that enables the scaling of relational pre-training to large-scale pseudo-labelled scene graph data.
> To enable fast scaling, RLIPv2 introduces Asymmetric Language-Image Fusion (ALIF), a mechanism that facilitates earlier and deeper gated cross-modal fusion with sparsified language encoding layers.
> ALIF leads to comparable or better performance than RLIPv1 in a fraction of the time for pre-training and fine-tuning.
> To obtain scene graph data at scale, we extend object detection datasets with free-form relation labels by introducing a captioner (\textit{e.g.,} BLIP) and a designed Relation Tagger.
> The Relation Tagger assigns BLIP-generated relation texts to region pairs, thus enabling larger-scale relational pre-training.
> Through extensive experiments conducted on Human-Object Interaction Detection and Scene Graph Generation, RLIPv2 shows state-of-the-art performance on three benchmarks under fully-finetuning, few-shot and zero-shot settings.
> Notably, the largest RLIPv2 achieves 23.29mAP on HICO-DET without any fine-tuning, yields 32.22mAP with just 1\% data and yields 45.09mAP with 100\% data.


## Todo list
Note that if you can not get access to the links provided below, try using another browser or contact me by e-mail. 
- [x] 🎉 Release code for pre-training, fine-tuning and inference.
- [ ] 🕘 Release pre-training and fine-tuning annotations. 
- [ ] 🕘 Release checkpoints for pre-training, few-shot, zero-shot and fine-tuning.  
- [ ] 🕘 Include support for inference on custom images.

Allow me to upload all the files in the following days to come because it is time-consuming.

## Information before using this repo
I changed all the paths to prevent from possible information leakage.
In order to run the code, you will need to configure the paths to match your own system.
To do this, search for the "/PATH/TO" placeholder in the code and replace it with the appropriate file path on your system. 
⭐⭐⭐Consider starring the repo! ⭐⭐⭐

## Environment setup
I recommend creating a new conda environment in order to run the code.
You can check `scripts/create_environment.txt` to acquire details on how to set up the environment.

## Model outline
This repo contains the implementation of various methods to resolve HOI detection (not limited to RLIP), aiming to serve as a benchmark for HOI detection. Below methods are included in this repo:
 - [RLIPv2-ParSeDA]() (model name in the repo: RLIP_ParSeDA_v2);
 - [RLIPv2-ParSeD]() (model name in the repo: RLIP_ParSeD_v2);
 - [RLIP-ParSe](https://arxiv.org/abs/2209.01814) (model name in the repo: RLIP-ParSe);
 - [ParSe](https://arxiv.org/abs/2209.01814) (model name in the repo: ParSe);
 - [RLIP-ParSeD](https://arxiv.org/abs/2209.01814) (model name in the repo: RLIP-ParSeD);
 - [ParSeD](https://arxiv.org/abs/2209.01814) (model name in the repo: ParSeD);
 - [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark) (model name in the repo: OCN), which is a prior work of RLIP;  
 - [QPIC](https://github.com/hitachi-rd-cv/qpic) (model name in the repo: DETRHOI);
 - [QAHOI](https://github.com/cjw2021/QAHOI) (model name in the repo: DDETRHOI);
 - [CDN](https://github.com/YueLiao/CDN) (model name in the repo: CDN);


## Citation
```bibtex
@inproceedings{Yuan2023RLIPv2,
  title={RLIPv2: Fast Scaling of Relational Language-Image Pre-training},
  author={Yuan, Hangjie and Zhang, Shiwei and Wang, Xiang and Albanie, Samuel and Pan, Yining and Feng, Tao and Jiang, Jianwen and Ni, Dong and Zhang, Yingya and Zhao, Deli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@inproceedings{Yuan2022RLIP,
  title={RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection},
  author={Yuan, Hangjie and Jiang, Jianwen and Albanie, Samuel and Feng, Tao and Huang, Ziyuan and Ni, Dong and Tang, Mingqian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{Yuan2022OCN,
  title={Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics},
  author={Hangjie Yuan and Mang Wang and Dong Ni and Liangpeng Xu},
  booktitle={AAAI},
  year={2022}
}
```


## Annotation preparation
| Dataset | Setting | Download |
| ---------- | :-----------:  | :-----------:  |
| VG | RLIP | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EWEPvw_EEttNt4TNHABDWbgB0S4LBPzlxvPidh_MhEEUTQ?e=j9gBjk) |
| COCO (pseudo) | RLIP | [Link]() |
| Objects365 (pseudo) | RLIP | [Link]() |
| Open Images | Fully-finetuning  | [Link]() |
| HICO-DET | Few-shot 1%, 10% | [Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/hj_yuan_zju_edu_cn/Eh7UufFbB_5Dutvr66g-t6sBn5wCeA0uzMwiy8mUxaD50g?e=IKB3SD) |
| HICO-DET | Zero-shot (UC-NF, UC-RF)\* | [Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/hj_yuan_zju_edu_cn/Ev9BzZxOlT5Mt04wOpIHA5kBP2eA6fijjweI_kh9WN3MUw?e=jMJmu6) |

Note: ① \* Zero-shot (NF) do not need any HICO-DET annotations for fine-tuning, so we only provide training annotations for the UC-NF and UC-RF setting.


## Pre-training datasets preparation

### 1. Visual Genome
Firstly, we could download VG dataset from the [official link](https://visualgenome.org/api/v0/api_home.html), inclduing images Part I and Part II. (**Note: If the official website is not working, you can use the link that I provide: [Images](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/Ed38rTcxgq9JnMdQS0SUSAIBE2azKnbq8_ZosJ6RZHaJjg?e=bpeuLt) and [Images2](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/Ea09ejSJ_KpJm_CKmmgMeScB81gSfJXD9gp7INzSrX53mg?e=pPQCo1).**) The annotations after pre-processing could be downloaded from the link above, which is used for pre-training. Note that this is generated from `scene_graphs.json` file by several pre-processing steps to remove redundant triplets. Also, several settings mentioned below also need the annotations that we provide. VG dataset and its corresponding annotations should be organized as follows:
```
VG
 |─ annotations
 |   |— scene_graphs_after_preprocessing.json
 |   :
 |— images
 |   |— 2409818.jpg
 |   |— n102412.jpg
 :   :
```

### 2. COCO
Firstly, try downloading the [COCO2017](https://cocodataset.org/#download) dataset from the official link. If you want to run R-Tagger, you need to download the bounding box annotations from the website as well. If you just want to perform relational pre-training, you can merely download the pseudo-annotations for COCO2017.
The dataset should be organized as follows:
```
COCO2017
 |— annotations
 |   |— instances_train2017.json
 |   |— instances_val2017.json
 |   └─ RLIPv2_train2017_threshold20.....json
 |   
 |— train2017
 |   |— 000000498666.jpg
 |   :
 |
 |— val2017
 |   |— 000000414261.jpg
 :   :
```

### 3. Objects365
Firstly, download the [Objects365](https://www.objects365.org/download.html) dataset from the official link. This dataset contains 51 training patches and 44 validation patches, which are summed to more than 1700k images used for pre-training. 
Similarly, if you want to run R-Tagger, you need to download the bounding box annotations from the website as well. 
If you just want to perform relational pre-training, you can merely download the pseudo-annotations for Objects365.
(Btw, you can try use the script in `scripts/datasets` folder.)
The dataset should be organized as follows:
```
Objects365
 |— train 
 |   |— patch0
 |   |— patch1
 |   :
 |   |— patch50
 |   └─ zhiyuan_objv2_train.json
 |
 |— val
 |   |— patch0
 |   |— patch1
 |   :
 |   |— patch43
 |   └─ zhiyuan_objv2_val.json
 |
 |— rel_annotations
 |    └─ RLIPv2_o365trainval_Tagger2.....json
 |
 └─ image_id_to_filepath.json
```


## Downstream dataset preparation
### 1. HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
qpic
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

### 2. V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
qpic
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
The annotation file has to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.

### 3. Open Images v6
Open Images v6 can be downloaded from this [link](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md).
We transform the annotations to the HICO-DET format, which can be downloaded from the link provided [above](## Annotation Preparation).
The dataset should be organized as follows:
```
Open Images v6
 |
 |─ images
 |    |─ ca5267a6336b71ea.jpg
 |    :
 |
 └─ annotations
```

