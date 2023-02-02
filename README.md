# ER-Test: Evaluating Explanation Regularization Methods for Language Models

This is the code for the paper titled 

This is the code and the dataset for the paper titled 

>[ER-Test: Evaluating Explanation Regularization Methods for Language Models. *Brihi Joshi\*, Aaron Chan\*, Ziyi Liu\*, Shaoliang Nie, Maziar Sanjabi, Hamed Firooz, Xiang Ren*](https://aclanthology.org/2022.findings-emnlp.242/)

accepted at [Findings of EMNLP 2022](https://2022.emnlp.org/).

If you end up using this code or the data, please cite our paper: 

```
=@inproceedings{joshi-etal-2022-er,
    title = "{ER}-Test: Evaluating Explanation Regularization Methods for Language Models",
    author = "Joshi, Brihi  and
      Chan, Aaron  and
      Liu, Ziyi  and
      Nie, Shaoliang  and
      Sanjabi, Maziar  and
      Firooz, Hamed  and
      Ren, Xiang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.242",
    pages = "3315--3336",
    abstract = "By explaining how humans would solve a given task, human rationales can provide strong learning signal for neural language models (NLMs). Explanation regularization (ER) aims to improve NLM generalization by pushing the NLM{'}s machine rationales (Which input tokens did the NLM focus on?) to align with human rationales (Which input tokens would humans focus on). Though prior works primarily study ER via in-distribution (ID) evaluation, out-of-distribution (OOD) generalization is often more critical in real-world scenarios, yet ER{'}s effect on OOD generalization has been underexplored.In this paper, we introduce ER-Test, a framework for evaluating ER models{'} OOD generalization along three dimensions: unseen datasets, contrast set tests, and functional tests. Using ER-Test, we comprehensively analyze how ER models{'} OOD generalization varies with the rationale alignment criterion (loss function), human rationale type (instance-level v/s task-level), number and choice of rationale-annotated instances, and time budget for rationale annotation. Across two tasks and six datasets, we show that ER has little impact on ID performance but yields large OOD performance gains, with the best ER criterion being task-dependent. Also, ER can improve OOD performance even with task-level or few human rationales. Finally, we find that rationale annotation is more time-efficient than label annotation for improving OOD performance. Our results with ER-Test help demonstrate ER{'}s utility and establish best practices for using ER effectively.",
}
```

# Quick Setup

## Requirements

- Python 3.5.x
To install the dependencies used in the code, you can use the __requirements.txt__ file as follows -

```
pip install -r requirements.txt
```

## Hydra working directory

Hydra will change the working directory to the path specified in `configs/hydra/default.yaml`. Therefore, if you save a file to the path `'./file.txt'`, it will actually save the file to somewhere like `logs/runs/xxxx/file.txt`. This is helpful when you want to version control your saved files, but not if you want to save to a global directory. There are two methods to get the "actual" working directory:

1. Use `hydra.utils.get_original_cwd` function call
2. Use `cfg.work_dir`. To use this in the config, can do something like `"${data_dir}/${.dataset}/${model.arch}/"`


## Config Key

- `work_dir` current working directory (where `src/` is)

- `data_dir` where data folder is

- `log_dir` where log folder is (runs & multirun)

- `root_dir` where the saved ckpt & hydra config are

## Offline mode
In offline mode, results are not logged to Neptune.
```
python main.py logger.offline=True
```

## Debug mode
In debug mode, results are not logged to Neptune, and we only train/evaluate for limited number of batches and/or epochs.
```
python main.py debug=True
```

# Example Commands

Here, we assume the following: 
- The `data_dir` is `../data`, which means `data_dir=${work_dir}/../data`.
- The dataset is `sst`.
- The attribution algorithm is `input-x-gradient`.

## 1. Build dataset
The commands below are used to build pre-processed datasets, saved as pickle files. The model architecture is specified so that we can use the correct tokenizer for pre-processing.

```
python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --split train --arch google/bigbird-roberta-base 

python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --split dev --arch google/bigbird-roberta-base 

python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --split test --arch google/bigbird-roberta-base 

```

If the dataset is very large, you have the option to subsample part of the dataset for smaller-scale experiements. For example, in the command below, we build a train set with only 1000 train examples (sampled with seed 0).
```
python scripts/build_dataset.py \
    --data_dir ../data \
    --dataset sst \
    --split train \
    --arch google/bigbird-roberta-base \
    --num_samples 1000 \
    --seed 0
```

## 2. Running ER Experiments

### A. Train Task LM without ER (No-ER setting in the paper)

The command below is the most basic way to run `main.py` and will train the Task LM without any explanation regularization (`model=lm`). 

However, since all models need to be evaluated w.r.t. explainability metrics, we need to specify an attribution algorithm for computing post-hoc explanations. This is done by setting `model.explainer_type=attr_algo` to specify that we are using an attribution algorithm based explainer (as opposed to `lm` or `self_lm`), `model.attr_algo` to specify the attribution algorithm, and `model.attr_pooling` to specify the attribution pooler.
```
python main.py -m \
    data=sst \
    model=lm \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

By default, checkpoints will not be saved (i.e., `save_checkpoint=False`), so you need to set `save_checkpoint=True` if you want to save the best checkpoint.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=lm \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

### B. Train Task LM with Explanation Regularization (ER)
We can also train the Task LM with ER (`model=expl_reg`). ER can be done using pre-annotated gold rationales or human-in-the-loop feedback.

Provide gold rationales for all train instances:
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.attr_algo=input-x-gradient \
    model.task_wt=1.0 \
    model.pos_expl_wt=0.5 \
    model.pos_expl_criterion=bce \
    model.neg_expl_wt=0.5 \
    model.neg_expl_criterion=l1 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```


