# Data-efficient Fine-tuning for LLM-based Recommendation
:bulb: This is the pytorch implementation of our paper 
> [Data-efficient Fine-tuning for LLM-based Recommendation](https://arxiv.org/pdf/2401.17197)

> [Xinyu Lin](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=0O_bs3UAAAAJ&sortby=pubdate), [Wenjie Wang](https://wenjiewwj.github.io/), [Yongqi Li](https://liyongqi67.github.io/), [Shuo Yang](https://shuoyang-1998.github.io/), [Fuli Feng](https://fulifeng.github.io/), [Yinwei Wei](https://weiyinwei.github.io/), [Tat-Seng Chua](https://www.chuatatseng.com/)

## Environment
- Anaconda 3 

Install the environment with the ``.yaml`` file and run
```
conda env create -f DEALRec.yaml
```

## Usage

### Data
The experimental data are in './data' folder, including Games, MicroLens-50K, and Book. 

### :red_circle: Pruning
The code for data pruning, including the score calculation and the coverage-enhanced sample selection is in './code/prune/'.
You can prune the data by running
```
python -u prune.py --data_name=$1 --model_name=$2 --lamda=$3 --k=$4 --log_name=$5 --gpu_id=$6
``` 
or use prune.sh
```
sh prune.sh <data_name> <surrogate_model_name> <lamda> <group_number> <log_name> <gpu_id>
```
- The selected samples' indices will be saved in './code/prune/selected/' folder.
- The explanation of hyper-parameters can be found in './code/prune/utils.py'. 
- The default hyper-parameter settings are detailed in './code/prune/hyper-parameters.txt'.

:star2: The surrogate model implemented here is SASRec. But it is highlighted that DEALRec is applicable to any other surrogate models, e.g., DCRec (refer to Section 4.3.2).

### :large_blue_circle: Few-shot Fine-tuning
Fine-tune LLM-based recommender model ([BIGRec](https://github.com/SAI990323/Grounding4Rec)) with few-shot samples obtained from pruning process.
The code for fine-tuning is in 'code/finetune/'. 
Fine-tune BIGRec with few-shot samples and get the results by running
```
sh finetune.sh <data_name> 
```

### :white_circle: Examples
1. Prune the data on Games
```
cd ./code/prune/
sh prune.sh games SASRec 0.3 50 log 0
```
2. Fine-tune BIGRec with few-shot samples (set at 1024 by default).
```
cd ./code/finetune/
sh finetune.sh games
```

### Evaluation
The codes and running scripts for evaluation are in 'code/finetune/data/' folder. 

## License

NUS © [NExT++](https://www.nextcenter.org/)