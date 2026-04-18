
import os
from datasets import load_dataset
from PIL import Image

def download_and_prepare_dataset(dataset_name='lambdalabs/pokemon-blip-captions', output_dir='data/pokemon'):
    print(f'Downloading dataset {dataset_name}...')
    dataset = load_dataset(dataset_name, split='train')
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    print(f'Saving {len(dataset)} images to {output_dir}...')
    for (i, item) in enumerate(dataset):
        image = item['image']
        text = item['text']
        image_path = os.path.join(output_dir, f'image_{i}.png')
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        image.save(image_path)
        metadata.append(f'{{"file_name": "image_{i}.png", "text": "{text}"}}')
        if (((i + 1) % 100) == 0):
            print(f'Processed {(i + 1)} images')
    with open(os.path.join(output_dir, 'metadata.jsonl'), 'w') as f:
        f.write('\n'.join(metadata))
    print(f'Dataset preparation complete. Saved to {output_dir}')
if (__name__ == '__main__'):
    download_and_prepare_dataset()
