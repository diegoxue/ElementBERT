---
language: en
tags: 
  - deberta
  - deberta-v3
  - fill-mask
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
license: mit
---

## DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks with 80GB training data. 

In [DeBERTa V3](https://arxiv.org/abs/2111.09543), we further improved the efficiency of DeBERTa using ELECTRA-Style pre-training with Gradient Disentangled Embedding Sharing. Compared to DeBERTa,  our V3 version significantly improves the model performance on downstream tasks.  You can find more technique details about the new model from our [paper](https://arxiv.org/abs/2111.09543).

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more implementation details and updates.

The DeBERTa V3 xsmall model comes with 12 layers and a hidden size of 384. It has only **22M** backbone parameters  with a vocabulary containing 128K tokens which introduces 48M parameters in the Embedding layer.  This model was trained using the 160GB data as DeBERTa V2.


#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 2.0 and MNLI tasks.

| Model             |Vocabulary(K)|Backbone #Params(M)| SQuAD 2.0(F1/EM) | MNLI-m/mm(ACC)|
|-------------------|----------|-------------------|-----------|----------|
| RoBERTa-base      |50     |86                 | 83.7/80.5 | 87.6/-   |
| XLNet-base        |32     |92                 | -/80.2    | 86.8/-   |
| ELECTRA-base      |30    |86                  | -/80.5    | 88.8/    |
| DeBERTa-base      |50     |100                |  86.2/83.1| 88.8/88.5|
| DeBERTa-v3-large|128|304                      | 91.5/89.0 | 91.8/91.9|
| DeBERTa-v3-base |128|86                       | 88.4/85.4 | 90.6/90.7|
| DeBERTa-v3-small  |128|44                     | 82.8/80.4 | 88.3/87.7|
| **DeBERTa-v3-xsmall** |128|**22**             | **84.8/82.0** | **88.1/88.3**|
| DeBERTa-v3-xsmall+SiFT|128|22                 | -/-       | 88.4/88.5|


[#| ELECTRA-small     |30    |9.6                | -         | -        |]::

#### Fine-tuning with HF transformers

```bash
#!/bin/bash

cd transformers/examples/pytorch/text-classification/

pip install datasets
export TASK_NAME=mnli

output_dir="ds_results"

num_gpus=8

batch_size=8

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 256 \
  --warmup_steps 1000 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 4.5e-5 \
  --num_train_epochs 3 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir $output_dir

```

### Citation

If you find DeBERTa useful for your work, please cite the following papers:

``` latex
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

``` latex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```
