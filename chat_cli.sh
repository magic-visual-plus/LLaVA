# python -m llava.serve.cli --model-path /opt/product/LLaVA/checkpoints/llava-llama-2-7b-chat-lightning-lora-preview --image-file "/opt/product/LLaVA/view.jpg" --model-base /opt/product/llama/llama-2-7b-chat-hf

# python -m llava.serve.cli \
#     --model-path /opt/product/LLaVA/checkpoints/llava-7b-llama-2-7b-chat \
#     --image-file "/opt/product/LLaVA/view.jpg" \
#     --load-4bit


# python -m llava.serve.cli \
#     --model-path /opt/product/LISA/checkpoints/LISA-7B-llava-llama-2-7b-chat-newly \
#     --image-file "/opt/product/LLaVA/view.jpg" \
#     --load-4bit --debug


python -m llava.serve.cli \
    --model-path /opt/product/LLaVA/checkpoints/llava-7b-llama-2-7b-chat \
    --image-file "/opt/product/LLaVA/img12.jpg" \
    --load-8bit