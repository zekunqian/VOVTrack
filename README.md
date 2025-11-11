## VOVTrack: Exploring the Potentiality in Raw Videos for Open-Vocabulary Multi-Object Tracking

### TL;DR

- VOVTrack training has two major parts:
  - DetPro-based detection training (Stage 1 + Stage 2 fine-tuning).
  - OVTrack-style tracker training (pair pretraining + self-supervised fine-tuning).
- All pretrained and trained weights are hosted on Hugging Face. Download them into {REPO_ROOT}:
  - `https://huggingface.co/clarkqian/VOVTrack/tree/main/saved_models`
- In this document, `{REPO_ROOT}` denotes your local project root directory.

---

## Overview

VOVTrack is trained in two parts:

1) DetPro Training:
- Stage 1: Standard DetPro training (same as the original [DetPro](https://github.com/dyabel/detpro)).
- Stage 2: Tracking-State-Aware Prompt-based Attention fine-tuning.

2) Tracker Training (OVTrack-style):
- Pair pretraining (same as [OVTrack](https://github.com/SysCV/ovtrack)).
- Self-supervised fine-tuning.

We place our modified DetPro source code under `{REPO_ROOT}/detpro`. For environment setup and dataset preparation, follow the original upstream projects (DetPro and OVTrack).

---

## Repository Structure (key parts)

- `detpro/` — Modified DetPro code.
- `ovtrack/` — OVTrack-style tracker training and evaluation code.
- `tools/` — Launcher scripts.
- `configs/` — Configuration files for VOVTrack association training.
- `detpro/configs` — Configuration files for VOVTrack detection training.
- `saved_models/` — Place checkpoints here after downloading from Hugging Face.

---

## Part 1: DetPro Training

### Code Location
- Modified source: `{REPO_ROOT}/detpro`

### Environment & Dataset
- Follow the original DetPro project strictly for environment and dataset preparation:
  - DetPro (official): `https://github.com/dyabel/detpro/tree/main`

### Stage 1: Standard DetPro Training

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./tools/dist_train.sh configs/lvis/detpro_ens_20e.py 4 \
  --work-dir vild_ens_20e_fg_bg_5_10_end \
  --cfg-options \
    model.roi_head.prompt_path={REPO_ROOT}/saved_models/pretrained_models/iou_neg5_ens.pth \
    model.roi_head.load_feature=True
```

Resulting checkpoint:
- `{REPO_ROOT}/saved_models/our_trained_models/detpro_stage1.pth`

### Stage 2: Tracking-State-Aware Prompt-based Attention Fine-tuning

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./tools/dist_train.sh configs/lvis/detpro_ens_20e_part_batch_prompt_reverse.py 4 \
  --work-dir workdirs/detpro_prompt_attention_finetune \
  --cfg-options \
    model.roi_head.prompt_path={REPO_ROOT}/saved_models/pretrained_models/iou_neg5_ens.pth  \
    model.roi_head.load_feature=True
```

Resulting checkpoint:
- `{REPO_ROOT}/saved_models/our_trained_models/detpro_stage2_finetune.pth`

---

## Part 2: OVTrack-style Tracker Training

### Environment Setup
- Please refer to the OVTrack installation guide: [OVTrack Installation](https://github.com/SysCV/ovtrack/blob/main/docs/INSTALL.md)

### Dataset
- Use datasets as in OVTrack. See: [OVTrack Get Started](https://github.com/SysCV/ovtrack/blob/main/docs/GET_STARTED.md)

### Stage 3: Pair Pretraining (same as OVTrack)

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
tools/dist_train.sh configs/ovtrack-teta/ovtrack_r50_reverse_without_inference.py 4 20956 \
  --work-dir work_dirs/VOVTrack_repair_base_cls/test \
  --cfg-options \
    total_epochs=10 \
    load_from={REPO_ROOT}/saved_models/our_trained_models/detpro_stage2_finetune.pth
```

Resulting checkpoint:
- `{REPO_ROOT}/saved_models/our_trained_models/ovtrack_pair.pth`

### Stage 4: Self-supervised Fine-tuning

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
tools/dist_train.sh configs/ovtrack-teta/adding_spatial/ovtrack_r50_self_train_fintune_adding_spatial_without_inference_ratio1.0.py 4 23532 \
  --work-dir work_dirs/VOVTrack_repair_base_cls/test \
  --cfg-options \
    model.roi_head.spatial_learning_rate=0.1 \
    load_from={REPO_ROOT}/saved_models/our_trained_models/ovtrack_pair.pth \
    total_epochs=20
```

Final checkpoint:
- `{REPO_ROOT}/saved_models/our_trained_models/ovtrack_finetune_final.pth`

---

## Inference on TAO Validation

Example command (use the final fine-tuned checkpoint under `{REPO_ROOT}`):


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
tools/dist_test_new.sh \
  configs/ovtrack-teta/adding_spatial/ovtrack_r50_self_train_fintune_adding_spatial_without_inference_ratio1.0.py \
  {REPO_ROOT}/saved_models/our_trained_models/ovtrack_finetune_final.pth \
  4 36141 --eval track \
  --eval-options resfile_path=results/vovtrack_iccv2025 \
  --cfg-options \
    model.tracker.match_score_thr=0.33 \
    model.test_cfg.rcnn.max_per_img=110 \
    model.roi_head.only_validation_categories=True \
    model.tracker.memo_frames=30 \
    model.tracker.momentum_embed=0.4
```

---

## Weights and Checkpoints

- All weights (pretrained and our trained checkpoints) are hosted here and should be placed under `{REPO_ROOT}`:
  - `https://huggingface.co/clarkqian/VOVTrack/tree/main/saved_models`
- Expected locations after download:
  - `{REPO_ROOT}/saved_models/pretrained_models/iou_neg5_ens.pth`
  - `{REPO_ROOT}/saved_models/pretrained_models/detpro_prompt.pt`
  - `{REPO_ROOT}/saved_models/pretrained_models/ovtrack_clip_distillation.pth`
  - `{REPO_ROOT}/saved_models/pretrained_models/ovtrack_detpro_prompt.pth`
  - `{REPO_ROOT}/saved_models/our_trained_models/detpro_stage1.pth`
  - `{REPO_ROOT}/saved_models/our_trained_models/detpro_stage2_finetune.pth`
  - `{REPO_ROOT}/saved_models/our_trained_models/ovtrack_pair.pth`
  - `{REPO_ROOT}/saved_models/our_trained_models/ovtrack_finetune_final.pth`

---

## Reproducibility Notes

- Match CUDA/cuDNN, PyTorch, MMCV/MMDetection, and other dependencies as recommended by the original DetPro and OVTrack projects.
- Follow OVTrack’s official docs for data preparation and splits.
- The provided commands assume 4 GPUs; adjust GPU count and batch sizes according to your hardware.

---

## Acknowledgements

- This repository builds upon and thanks the following excellent projects:
  - OVTrack: `https://github.com/SysCV/ovtrack`
  - DetPro: `https://github.com/dyabel/detpro`

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{qian2025vovtrack,
  title={VOVTrack: Exploring the Potentiality in Raw Videos for Open-Vocabulary Multi-Object Tracking},
  author={Qian, Zekun and Han, Ruize and Hou, Junhui and Song, Linqi and Feng, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7472--7482},
  year={2025}
}
```

---

## License

This project is released under a research-friendly license. Please also check the upstream licenses of OVTrack and DetPro for their respective terms.


