#!/bin/bash

set -euo pipefail

# set path firstly
lightx2v_path=D:/yanzhang/repo/LightX2V
model_path=D:/yanzhang/models/Wan2.1-T2V-1.3B
python_bin=C:/Users/arda/miniforge3/envs/yanzhang-lightx2v/python.exe

export CUDA_VISIBLE_DEVICES=0
export PLATFORM=intel_xpu

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

logs_dir=${lightx2v_path}/save_results/stability_logs
outputs_dir=${lightx2v_path}/save_results/stability_outputs

mkdir -p "${logs_dir}" "${outputs_dir}"

negative_prompt="camera shake, oversaturated, overexposed, static frame, blurry details, subtitles, watermark, low quality, ugly, deformed, extra fingers, bad hands, bad face, malformed limbs, cluttered background"

run_case() {
    local case_name="$1"
    local config_name="$2"
    local prompt="$3"
    local output_name="$4"
    local log_name="$5"

    local config_path="${lightx2v_path}/configs/platforms/intel_xpu/${config_name}"
    local output_path="${outputs_dir}/${output_name}"
    local log_path="${logs_dir}/${log_name}"

    echo "===============================================================================" | tee "${log_path}"
    echo "START $(date '+%Y-%m-%d %H:%M:%S') | ${case_name}" | tee -a "${log_path}"
    echo "CONFIG: ${config_path}" | tee -a "${log_path}"
    echo "OUTPUT: ${output_path}" | tee -a "${log_path}"
    echo "PROMPT: ${prompt}" | tee -a "${log_path}"
    echo "===============================================================================" | tee -a "${log_path}"

    "${python_bin}" -m lightx2v.infer \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${output_path}" 2>&1 | tee -a "${log_path}"

    echo "END $(date '+%Y-%m-%d %H:%M:%S') | ${case_name}" | tee -a "${log_path}"
    echo | tee -a "${log_path}"
}

run_case \
"train_full" \
"wan_t2v_1_3_teacache_hidden.json" \
"A cinematic aerial shot of a sleek futuristic train gliding through snowy mountains at sunrise." \
"train_full.mp4" \
"train_full.log"

run_case \
"train_block0" \
"wan_t2v_1_3_teacache_hidden_embed0_block0.json" \
"A cinematic aerial shot of a sleek futuristic train gliding through snowy mountains at sunrise." \
"train_block0.mp4" \
"train_block0.log"

run_case \
"chef_full" \
"wan_t2v_1_3_teacache_hidden.json" \
"A street food chef flips noodles over a roaring flame in a busy night market, sparks flying everywhere." \
"chef_full.mp4" \
"chef_full.log"

run_case \
"chef_block0" \
"wan_t2v_1_3_teacache_hidden_embed0_block0.json" \
"A street food chef flips noodles over a roaring flame in a busy night market, sparks flying everywhere." \
"chef_block0.mp4" \
"chef_block0.log"

run_case \
"boat_full" \
"wan_t2v_1_3_teacache_hidden.json" \
"A small sailboat battles dramatic ocean waves during a stormy sunset, with sea spray and fast-moving clouds." \
"boat_full.mp4" \
"boat_full.log"

run_case \
"boat_block0" \
"wan_t2v_1_3_teacache_hidden_embed0_block0.json" \
"A small sailboat battles dramatic ocean waves during a stormy sunset, with sea spray and fast-moving clouds." \
"boat_block0.mp4" \
"boat_block0.log"
