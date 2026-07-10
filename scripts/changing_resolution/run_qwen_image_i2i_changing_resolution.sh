#!/bin/bash

lightx2v_path=${lightx2v_path:-D:/yanzhang/repo/LightX2V}
model_path=${model_path:-D:/yanzhang/models/Qwen-Image-Edit-2511-bnb-nf4}
image_path=${image_path:-D:/yanzhang/project/qwen-image-edit-finetune-skill-test/train_dataset/control_images/sample_000.png}
prompt_file=${prompt_file:-D:/yanzhang/project/qwen-image-edit-finetune-skill-test/train_dataset/training_images/sample_000.txt}
target_shape=${target_shape:-256 256}
if [ -z "${save_result_path}" ]; then
    save_result_path="${lightx2v_path}/save_results/output_lightx2v_qwen_image_i2i_changing_resolution.png"
fi

if [ -z "${prompt}" ]; then
    if [ -f "${prompt_file}" ]; then
        prompt=$(cat "${prompt_file}")
    else
        prompt="Convert the image to anime style"
    fi
fi

export PLATFORM=${PLATFORM:-intel_xpu}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ "${RUN_QWEN_IMAGE_EDIT_FULL}" != "1" ]; then
    cat <<EOF
Refusing to launch full Qwen-Image-Edit by default because it can exhaust system RAM/XPU memory.

For a guarded small smoke run, execute:
  RUN_QWEN_IMAGE_EDIT_FULL=1 target_shape="256 256" infer_steps_override=3 changing_steps_override="1,2" sh $0

For a real run, set RUN_QWEN_IMAGE_EDIT_FULL=1 explicitly and make sure enough memory is available.
EOF
    exit 2
fi

source "${lightx2v_path}/scripts/base/base.sh"

config_json=${config_json:-"${lightx2v_path}/configs/changing_resolution/qwen_image_i2i_2511_U.json"}
if [ -n "${infer_steps_override}" ] || [ -n "${changing_steps_override}" ] || [ -n "${changing_resolution_override}" ]; then
    tmp_config="${TMPDIR:-/tmp}/qwen_image_i2i_changing_resolution_${RANDOM}.json"
    python - "${config_json}" "${tmp_config}" "${infer_steps_override}" "${changing_steps_override}" "${changing_resolution_override}" <<'PY'
import json
import sys

src, dst, infer_steps, changing_steps, changing_resolution = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)
if infer_steps:
    config["infer_steps"] = int(infer_steps)
if changing_steps:
    config["changing_resolution_steps"] = [int(item) for item in changing_steps.split(",") if item]
if changing_resolution:
    value = changing_resolution.strip().lower()
    if value in ("1", "true", "yes", "on"):
        config["changing_resolution"] = True
    elif value in ("0", "false", "no", "off"):
        config["changing_resolution"] = False
    else:
        raise ValueError(f"Invalid changing_resolution_override={changing_resolution!r}; use true or false.")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
PY
    config_json="${tmp_config}"
fi

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --prompt "${prompt}" \
    --negative_prompt " " \
    --image_path "${image_path}" \
    --save_result_path "${save_result_path}" \
    --target_shape ${target_shape} \
    --seed 0
