#!/bin/bash

echo "--- Starting downloads and shit ---"
cd /workspace/ComfyUI/custom_nodes/ &&


git clone https://github.com/kijai/ComfyUI-SolAttn_triton &&
git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo &&
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 &&
git clone https://github.com/CherrySkyArt/cherryskyart_comfyui_nodes &&
git clone https://github.com/CherrySkyArt/text_editor_minimax &&
git clone https://github.com/crystian/ComfyUI-Crystools &&
git clone https://github.com/StableLlama/ComfyUI-basic_data_handling.git &&
git clone https://github.com/munkyfoot/ComfyUI-TextOverlay.git &&
git clone https://github.com/christian-byrne/audio-separation-nodes-comfyui.git &&
git clone https://github.com/chrisgoringe/cg-use-everywhere.git &&
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git &&
git clone https://github.com/1038lab/ComfyUI-RMBG.git &&
git clone https://github.com/yolain/ComfyUI-Easy-Use.git &&
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git &&
git clone https://github.com/kijai/ComfyUI-KJNodes.git &&
git clone https://github.com/rgthree/rgthree-comfy.git &&
git clone https://github.com/diodiogod/TTS-Audio-Suite.git &&
git clone https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git &&
git clone https://github.com/geroldmeisinger/ComfyUI-outputlists-combiner.git &&



for dir in */; do
    req="${dir}requirements.txt"
    if [ -f "$req" ]; then
        echo "Installing requirements for $dir"
        pip install -r "$req"
    else
        echo "No requirements.txt for $dir — skipping"
    fi
done



ARIA2_OPTS="-c -x 8 -s 8 -k 1M --file-allocation=none"
sudo apt update && sudo apt install -y aria2 

cd ../models


cd vae
aria2c $ARIA2_OPTS -o minimax_h3_video_vae_fp16.safetensors "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" &
aria2c $ARIA2_OPTS -o minimax_h3_audio_vae_fp32.safetensors "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" &

cd ../text_encoders
aria2c $ARIA2_OPTS -o qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" &

cd ../loras
aria2c $ARIA2_OPTS -o drbaph----minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors.safetensors "https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors" &

cd ../diffusion_models
aria2c $ARIA2_OPTS -o 10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors "https://huggingface.co/TenStrip/10Eros-Max/resolve/main/10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors" & 

cd ../loras
aria2c $ARIA2_OPTS -o boobs---H3_slop_bounce.safetensors	                 "https://huggingface.co/DasCaptain/nsfw_models/resolve/main/boobs---H3_slop_bounce.safetensors" &
aria2c $ARIA2_OPTS -o boobs---MiniMax3_Boob%20Physics_v1.safetensors	  	 "https://huggingface.co/DasCaptain/nsfw_models/resolve/main/boobs---MiniMax3_Boob%20Physics_v1.safetensors" &
aria2c $ARIA2_OPTS -o boobs---bounceV07-000230_Intense---500mb.safetensors	 "https://huggingface.co/DasCaptain/nsfw_models/resolve/main/boobs---bounceV07-000230_Intense---500mb.safetensors" &
aria2c $ARIA2_OPTS -o boobs---breastplayjiggle_h3_v1.safetensors	 		 "https://huggingface.co/DasCaptain/nsfw_models/resolve/main/boobs---breastplayjiggle_h3_v1.safetensors" &

cd ../diffusion_models 
aria2c $ARIA2_OPTS -o minimax_h3_ref2va_pruned_int8_convrot_MSCOPE.safetensors "https://modelscope.ai/models/Comfy-Org/MiniMax-H3/resolve/master/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" &

wait
echo "All downloads complete"