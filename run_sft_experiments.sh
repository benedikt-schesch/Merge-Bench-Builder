#!/bin/bash

# Configuration arrays
LR=(1e-3 1e-4 1e-5)
WEIGHT_DECAY=(0 0.01)
SCHEDULER=("linear" "cosine")
EPOCHS=(1 3)
USE_GPUS=(2 3 4 5 6)

# Counter for GPU assignment
gpu_index=0
job_count=0

# Function to format learning rate for folder names
format_lr() {
    local lr=$1
    case $lr in
        "1e-3") echo "0.001" ;;
        "1e-4") echo "0.0001" ;;
        "1e-5") echo "1e-05" ;;  # Keep scientific notation as seen in examples
        *) echo $lr ;;
    esac
}

# Function to format weight decay for folder names
format_wd() {
    local wd=$1
    if [ "$wd" == "0" ]; then
        echo "0.0"
    else
        echo $wd
    fi
}

# Function to wait for background jobs if we've reached GPU limit
wait_for_gpu() {
    if [ $job_count -ge ${#USE_GPUS[@]} ]; then
        wait -n  # Wait for any background job to finish
        ((job_count--))
    fi
}

# Function to wait for evaluation jobs (4 jobs per GPU)
wait_for_eval_gpu() {
    local max_jobs=$((${#USE_GPUS[@]} * 4))  # 4 jobs per GPU
    if [ $eval_job_count -ge $max_jobs ]; then
        wait -n  # Wait for any background job to finish
        ((eval_job_count--))
    fi
}

echo "Starting SFT training with all parameter combinations..."
echo "Total GPUs available: ${#USE_GPUS[@]}"

# Array to store all model directories for evaluation
model_dirs=()

# Generate all combinations and run training
for lr in "${LR[@]}"; do
    for wd in "${WEIGHT_DECAY[@]}"; do
        for sched in "${SCHEDULER[@]}"; do
            for epochs in "${EPOCHS[@]}"; do
                # Format parameters for folder name
                lr_formatted=$(format_lr $lr)
                wd_formatted=$(format_wd $wd)

                # Check if output folder already exists
                output_dir="outputs/unsloth/DeepSeek-R1-Distill-Qwen-14B/sft_model_lr${lr_formatted}_epochs${epochs}_wd${wd_formatted}_${sched}/final_model"

                # Add to model directories for evaluation
                model_dirs+=("$output_dir")

                if [ -d "$output_dir" ]; then
                    echo "Skipping training: LR=$lr, WD=$wd, Scheduler=$sched, Epochs=$epochs - output already exists"
                    continue
                fi

                # Get current GPU
                current_gpu=${USE_GPUS[$gpu_index]}

                echo "Starting training: LR=$lr, WD=$wd, Scheduler=$sched, Epochs=$epochs on GPU $current_gpu"

                # Run training in background on specific GPU
                CUDA_VISIBLE_DEVICES=$current_gpu python3 sft_train.py \
                    --lr $lr \
                    --weight_decay $wd \
                    --lr_scheduler_type $sched \
                    --epochs $epochs &

                # Update counters
                ((job_count++))
                ((gpu_index++))

                # Reset GPU index if we've used all GPUs
                if [ $gpu_index -ge ${#USE_GPUS[@]} ]; then
                    gpu_index=0
                fi

                # Wait if we've filled all GPUs
                wait_for_gpu

                # Small delay to avoid overwhelming the system
                sleep 2
            done
        done
    done
done

# Wait for all remaining training jobs to complete
echo "Waiting for all training jobs to complete..."
wait

echo "All SFT training jobs completed!"

# Reset counters for evaluation
eval_job_count=0
eval_gpu_index=0

echo ""
echo "Starting model evaluation..."
echo "Total models to evaluate: ${#model_dirs[@]}"
echo "Running up to 4 evaluations per GPU"

# Run evaluation for all models
for model_dir in "${model_dirs[@]}"; do
    # Check if model directory exists
    if [ ! -d "$model_dir" ]; then
        echo "Skipping evaluation: Model directory $model_dir does not exist"
        continue
    fi

    # Get current GPU (cycling through available GPUs)
    current_gpu=${USE_GPUS[$eval_gpu_index]}

    echo "Starting evaluation for: $model_dir on GPU $current_gpu"

    # Run evaluation in background on specific GPU
    CUDA_VISIBLE_DEVICES=$current_gpu python3 eval.py --model_name "$model_dir" &

    # Update counters
    ((eval_job_count++))
    ((eval_gpu_index++))

    # Reset GPU index if we've used all GPUs
    if [ $eval_gpu_index -ge ${#USE_GPUS[@]} ]; then
        eval_gpu_index=0
    fi

    # Wait if we've filled all GPU slots (4 jobs per GPU)
    wait_for_eval_gpu

    # Small delay to avoid overwhelming the system
    sleep 1
done

# Wait for all remaining evaluation jobs to complete
echo "Waiting for all evaluation jobs to complete..."
wait

echo "All evaluations completed!"
echo "Summary:"
echo "- Training jobs: Completed for ${#model_dirs[@]} model configurations"
echo "- Evaluation jobs: Completed for all available models"
