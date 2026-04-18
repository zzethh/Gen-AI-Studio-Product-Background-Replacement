
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import mlflow
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def train_lora(args):
    hf_token = os.environ.get('HF_TOKEN', None)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    try:
        mlflow.start_run()
        mlflow.log_params(vars(args))
    except Exception as e:
        print(f'MLflow start failed: {e}')
    model_id = args.model_id
    print(f'Loading inpainting model: {model_id}')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None, use_auth_token=hf_token).to('cuda')
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print('Xformers enabled.')
    except:
        pass
    unet.enable_gradient_checkpointing()
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'], lora_dropout=0.1)
    unet = get_peft_model(unet, lora_config)
    unet.train()
    dataset = load_dataset(args.dataset_name, split='train', token=hf_token)
    if (args.max_train_samples is not None):
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    train_transforms = transforms.Compose([transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR), transforms.RandomCrop(args.resolution), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess(examples):
        images = [train_transforms(img.convert('RGB')) for img in examples[args.image_column]]
        input_ids = tokenizer(examples[args.caption_column], max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return {'pixel_values': images, 'input_ids': input_ids}
    dataset.set_transform(preprocess)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(filter((lambda p: p.requires_grad), unet.parameters()), lr=args.learning_rate)
    noise_scheduler = pipe.scheduler
    global_step = 0
    for epoch in range(args.num_epochs):
        for (step, batch) in enumerate(train_dataloader):
            pv = batch['pixel_values']
            pixel_values = (torch.stack(pv) if isinstance(pv, list) else pv)
            pixel_values = pixel_values.to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents * vae.config.scaling_factor)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            mask = torch.zeros(bsz, 1, latents.shape[2], latents.shape[3], device='cuda')
            masked_image_latents = torch.zeros_like(latents)
            unet_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
            encoder_hidden_states = text_encoder(input_ids)[0]
            model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
            loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction='mean')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            try:
                mlflow.log_metric('loss', loss.item(), step=global_step)
            except:
                pass
            if ((global_step % 10) == 0):
                print(f'Epoch {epoch} | Step {global_step} | Loss: {loss.item()}')
            global_step += 1
            if (global_step >= args.max_steps):
                break
        if (global_step >= args.max_steps):
            break
    unet.save_pretrained(args.output_dir)
    try:
        mlflow.log_artifact(args.output_dir)
    except:
        pass
    print(f'LoRA models saved to {args.output_dir}')
    try:
        mlflow.end_run()
    except:
        pass
if (__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-inpainting')
    parser.add_argument('--dataset_name', type=str, default='ashraq/fashion-product-images-small')
    parser.add_argument('--image_column', type=str, default='image')
    parser.add_argument('--caption_column', type=str, default='productDisplayName')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='models/fashion-lora')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--experiment_name', type=str, default='LoRA-Fashion-Inpainting')
    args = parser.parse_args()
    train_lora(args)
