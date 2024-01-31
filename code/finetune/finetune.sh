accelerate config

dataset=$1
fewshot=1024

# generate data for LLM-based recommender models (BIGRec)
cd data/
sh gen_fewshot.sh $dataset $fewshot
cd ../

for seed in 2023
do
    for sample in 1024
    do  
        # few-shot fine-tuning
        echo "seed: $seed"
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune.py \
            --base_model /your/path/to/llama/hf/model/ \
            --train_data_path ./data/${dataset}/train/train-${sample}.json \
            --val_data_path ./data/${dataset}/valid/valid-${sample}.json \
            --output_dir ./model/${dataset}/${seed}_${sample} \
            --batch_size 128 \
            --micro_batch_size 16 \
            --num_epochs 50 \
            --learning_rate 1e-4 \
            --cutoff_len 512 \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_target_modules '[q_proj,v_proj]' \
            --train_on_inputs \
            --group_by_length \
            --resume_from_checkpoint '/your/path/to/alpaca-lora-7B/' \
            --seed $seed \
            --sample $sample
        
        # generate
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=16253 inference_ddp.py \
            --lora_weights ./model/${dataset}/${seed}_${sample} \
            --result_json_data ./results/${dataset}/${seed}_${sample}.json;

        # evaluate
        cd data/
        PWD=$(pwd)
        echo "current work directory: $PWD"

        gpu_id=0
        res_file=${seed}_${sample}
        sh evaluate.sh ${res_file} ${gpu_id}
    done
done