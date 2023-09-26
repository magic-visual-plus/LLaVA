import pytest
from loguru import logger
import os
import warnings
import shutil
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

def test_clip_load():
    # vision_tower_name = 'openai/clip-vit-large-patch14'
    # device_map = f'cuda:1'
    # device_map = 'cpu'
    # device_map = 'auto'
    device_id = 1
    torch.cuda.set_device(device_id)
    device_map = {"": torch.cuda.current_device()}
    vision_tower_name = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name, device_map=device_map)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, device_map=device_map)
    vision_tower.requires_grad_(False)
    
    image_file = '/opt/product/LLaVA/view.jpg'
    image = Image.open(image_file).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt").to(torch.cuda.current_device())
    for i in range(100):
        outputs = vision_tower(**inputs, output_hidden_states=True)

    select_layer = -2
    # 使用倒数第二层当作 embindings 使用
    image_features = outputs.hidden_states[select_layer]
    # torch.Size([1, 257, 1024])
    print(image_features.shape)
    image_features = image_features[:, 1:]
    # torch.Size([1, 256, 1024])
    print(image_features.shape)

def test_load_lora():
    model_path = "/opt/product/LLaVA/checkpoints/llava-llama-2-7b-chat-lightning-lora-preview"
    model_base = "/opt/product/llama/llama-2-7b-chat-hf"
    model_name = get_model_name_from_path(model_path)
    logger.info("model_name {}", model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device_map='cpu')
    logger.info("model {}", model.keys())
    logger.info("context len {}", context_len)
    ...

def test_peft_model_load():
    model_path = "/opt/product/LLaVA/checkpoints/llava-llama-2-7b-chat-lightning-lora-preview"
    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    
    model_base = "/opt/product/llama/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    ...
    
if __name__ == "__main__":
    # test_peft_model_load()
    # test_load_lora()
    test_clip_load()