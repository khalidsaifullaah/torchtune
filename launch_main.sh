# launch with:
# launch launch_main.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30
# or
# launch launch_main.sh --classical_logfile_names --gpu_type rtxa6000 --mem 30 --qos vulcan-high_long

tune run lora_finetune_single_device --config recipes/configs/guardian_models/qwen_2_5_05B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_1B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_3B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_7B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/qwen_2_5_14B_lora_multirule_v2.yaml

tune run lora_finetune_single_device --config recipes/configs/guardian_models/8B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/WildGuard_7B_lora_7500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/WildGuard_7B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/GuardReasoner_8B_lora_7500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/GuardReasoner_8B_lora_multirule_v2.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/LlamaGuard_8B_lora_7500.yaml
tune run lora_finetune_single_device --config recipes/configs/guardian_models/LlamaGuard_8B_lora_multirule_v2.yaml

# tune run lora_finetune_single_device --config recipes/configs/guardian_models/qwen_2_5_05B_lora_7500.yaml
# tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_1B_lora_7500.yaml
# tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_3B_lora_7500.yaml
# tune run lora_finetune_single_device --config  recipes/configs/guardian_models/qwen_2_5_7B_lora_7500.yaml
# tune run lora_finetune_single_device --config recipes/configs/guardian_models/qwen_2_5_14B_lora_7500.yaml