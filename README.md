## Overview

This repository consists of the following Python scripts:

**Teacher training**
- `scripts/train_teacher_jets.py` – Train a tabular teacher (JetClass).
- `scripts/train_teacher_cifar.py` – Train a CIFAR-10 teacher.
- `scripts/train_teacher_audio.py` – Train an audio teacher (UrbanSound8K).

**Student training**
- `scripts/train_student_jets.py`, `scripts/train_student_cifar.py`, `scripts/train_student_audio.py` – Train a student from a teacher with our method (`ugkd`), one of the baselines, or one of the fused methods.

**Data and utilities**
- `scripts/make_jets_data.py`, `scripts/make_cifar_data.py`, `scripts/make_audio_data.py` – Build the paper's partially-expert-teacher datasets from the raw data.
- `scripts/evaluate.py` – Evaluate a trained teacher or student.

**Synthetic experiments**
- `appendix/theory_experiments.py` – The synthetic experiments of the theory appendix.
- `appendix/proxy_experiments.py` – The synthetic experiments relating aleatoric and epistemic uncertainty to BCP diversity and teacher error.

## Credits and Citation

The baselines and the Laplace approximation are implemented based on their papers and, where available, their released code; the CIFAR-10-C corruptions are ported from the code of Hendrycks and Dietterich. If you find this repository useful in your research, please consider citing the works our code builds upon:

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}

@inproceedings{zhao2022decoupled,
  title={Decoupled knowledge distillation},
  author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11953--11962},
  year={2022}
}

@inproceedings{wang2025abkd,
  title={{ABKD}: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via $\alpha$-$\beta$-Divergence},
  author={Guanghui Wang and Zhiyong Yang and Zitai Wang and Shi Wang and Qianqian Xu and Qingming Huang},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@inproceedings{sun2024logit,
  title={Logit standardization in knowledge distillation},
  author={Sun, Shangquan and Ren, Wenqi and Li, Jingzhi and Wang, Rui and Cao, Xiaochun},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={15731--15740},
  year={2024}
}

@inproceedings{sun2026uncertainty,
  title={Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models},
  author={Sun, Jingchen and Han, Shaobo and Patel, Deep and Kohno, Wataru and Jin, Can and Chen, Changyou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@article{guo2024leveraging,
  title={Leveraging logit uncertainty for better knowledge distillation},
  author={Guo, Zhen and Wang, Dong and He, Qiang and Zhang, Pengzhou},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={31249},
  year={2024}
}

@article{hemmatian2024uncertainty,
  title={Uncertainty-based knowledge distillation for {B}ayesian deep neural network compression},
  author={Hemmatian, Mina and Shahzadi, Ali and Mozaffari, Saeed},
  journal={International Journal of Approximate Reasoning},
  volume={175},
  pages={109301},
  year={2024}
}

@inproceedings{zheng2024knowledge,
  title={Knowledge distillation based on transformed teacher matching},
  author={Zheng, Kaixiang and Yang, En-Hui},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@article{mishra2026correctness,
  title={Correctness-Aware Knowledge Distillation for Enhanced Student Learning},
  author={Mishra, Ishan and Mishra, Deepak and Xiong, Jinjun},
  journal={Transactions on Machine Learning Research},
  year={2026}
}

@inproceedings{zhou2021rethinking,
  title={Rethinking Soft Labels for Knowledge Distillation: A Bias--Variance Tradeoff Perspective},
  author={Zhou, Helong and Song, Liangchen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{hu2024less,
  title={Less or more from teacher: Exploiting trilateral geometry for knowledge distillation},
  author={Hu, Chengming and Wu, Haolun and Li, Xuan and Ma, Chen and Chen, Xi and Wang, Boyu and Yan, Jun and Liu, Xue},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{lu2021rw,
  title={{RW-KD}: Sample-wise loss terms re-weighting for knowledge distillation},
  author={Lu, Peng and Ghaddar, Abbas and Rashid, Ahmad and Rezagholizadeh, Mehdi and Ghodsi, Ali and Langlais, Philippe},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={3145--3152},
  year={2021}
}

@inproceedings{hendrycks2018benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@article{daxberger2021laplace,
  title={Laplace redux-effortless {B}ayesian deep learning},
  author={Daxberger, Erik and Kristiadi, Agustinus and Immer, Alexander and Eschenhagen, Runa and Bauer, Matthias and Hennig, Philipp},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20089--20103},
  year={2021}
}
```

## Install

```
pip install -r requirements.txt
pip install -e .
```

## Teacher Training

Scripts: `train_teacher_jets.py`, `train_teacher_cifar.py`, `train_teacher_audio.py`.
A teacher script trains the network on its training data, fits a last-layer Laplace posterior on it, and saves both to the file given by `--out`.

**Common parameters**

The following arguments can be used in all training scripts, see `--help` for the defaults of each script:

- `--model` – architecture (see Models below)
- `--optimizer` – `sgd` or `adam`
- `--lr` – initial learning rate
- `--momentum` – momentum (sgd only)
- `--weight-decay` – weight decay
- `--epochs` – number of training epochs
- `--batch-size` – training batch size
- `--schedule` – `constant` or `cosine` learning-rate schedule
- `--label-smoothing` – label smoothing
- `--seed` – random seed
- `--device` – `auto`, `cpu` or `cuda`

**Posterior parameters**

- `--laplace` – `exact` or `kfac` (our implementation), or `library-full` / `library-kron` (aleximmer/Laplace)
- `--tau` – prior precision, a number or `marglik` (fitted by marginal likelihood)

**Inputs**

- jets: `--train` – an `.npz` with arrays `x` (N, d) and `y` (N,)
- cifar: `--data-root` – a torchvision CIFAR-10 folder (downloaded if absent), with `--teacher-frac` the share of the training images the teacher sees; or `--data` – an `.npz` with `x` uint8 (N, 32, 32, 3) and `y`
- audio: `--train-csv` – a CSV with columns `path,label` and `--root` the folder of the clips; or `--data` – an `.npz` of log-mel spectrograms


## Student Training

Scripts: `train_student_jets.py`, `train_student_cifar.py`, `train_student_audio.py`.
A student script takes a teacher file, the student's training data, an architecture and a distillation method, trains the student and saves it to `--out`. The common training parameters above apply. Additional parameters:

- `--teacher` – the teacher file (required)
- `--method` – the distillation method (see Methods below)
- `--method-arg KEY=VALUE` – override one of the method's own parameters (repeatable)
- `--mc-samples` – number of posterior draws per teacher read
- `--temperature` – distillation temperature
- `--val-frac` – share of the training data held out for methods that need a validation batch (`tgeo`, `rwkd`)


## Evaluation

```
python scripts/evaluate.py --model student.pt --data test.npz
```

Prints accuracy and negative log-likelihood, and accuracy per group when the data file carries a `group` array.

## Methods

| method | parameters and defaults |
|---|---|
| `ugkd` | - |
| `hinton` | lambda 1 (0 = labels only) |
| `lud` | - |
| `mcd` | alpha 12, beta 2 |
| `ubkd` | a0 0.95, beta_a 0.6 or `auto`, beta_T 0, anchor_q 50 |
| `dkd` | alpha 1, beta 8, ce_weight 1, warmup_frac 20/240 |
| `betakd` | init_beta 1 |
| `tgeo` | hidden (32, 16), ctrl_lr 0.1, ctrl_wd 0.05 |
| `wls` | alpha 2.25 |
| `rwkd` | delta 1e-8, beta 1 |
| `abkd` | alpha 0.9, beta 0.2, ce_weight 1, kd_weight 32 |
| `lsd` | ce_weight 0.1, kd_weight 9 |
| `wttm` | lambda 0.5, ce_weight 1 |
| `dkd_ugkd`, `abkd_ugkd`, `lsd_ugkd`, `wttm_ugkd`, `wls_ugkd` | the base method's parameters |

## Models

| modality | names |
|---|---|
| jets | `mlp_256`, `mlp_64` |
| cifar | `convnet6`, `convnet3`, `resnet20`, `resnet32`, `wrn_16_2`, `wrn_16_1`, `wrn_40_2`, `wrn_40_1`, `shufflenet_v1` |
| audio | `cnn8`, `cnn8n`, `resnet14`, `cnn4`, `cnn4n`, `resnet8` |

## Synthetic Experiments

```
python appendix/theory_experiments.py
python appendix/proxy_experiments.py
```

Each writes its results to `appendix/results/`.
