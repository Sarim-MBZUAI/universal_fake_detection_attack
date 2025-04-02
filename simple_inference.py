import os
import torch
from PIL import Image
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import numpy as np
from networks.trainer import Trainer
from options.train_options import TrainOptions

def main():
    # Use the existing option parser to maintain compatibility
    opt = TrainOptions().parse()
    opt.isTrain = False
    
    # Define checkpoint path
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'model_epoch_best.pth')
    real_dir = '/l/users/sarim.hashmi/Thesis/NIPS/Defactify4_Train/Test_set/real'
    fake_dir = '/l/users/sarim.hashmi/Thesis/NIPS/Defactify4_Train/Test_set/fake'
    
    # Load model
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
    print(f"Using device: {device}")
    
    trainer = Trainer(opt)
    
    # Load checkpoint properly - extract the model state from the full checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if checkpoint contains 'model' key (full training state)
    if 'model' in checkpoint:
        trainer.model.load_state_dict(checkpoint['model'])
    else:
        trainer.model.load_state_dict(checkpoint)
    
    trainer.model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((opt.cropSize, opt.cropSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process real and fake directories
    process_directory(trainer.model, real_dir, transform, device, "real")
    process_directory(trainer.model, fake_dir, transform, device, "fake")

def process_directory(model, directory, transform, device, dir_type):
    image_paths = glob(os.path.join(directory, '*.jpg'))
    results = []
    
    print(f"\nProcessing {len(image_paths)} images from {dir_type} directory...")
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            # Handle output whether it's a tensor or tuple
            if isinstance(output, tuple):
                output = output[0]
            prob = torch.sigmoid(output).item()
            
        pred = "Real" if prob >= 0.5 else "Fake"
        confidence = prob if pred == "Real" else 1 - prob
        is_correct = (pred == "Real" and dir_type == "real") or (pred == "Fake" and dir_type == "fake")
        
        results.append({
            'path': img_path,
            'prediction': pred,
            'confidence': confidence,
            'raw_score': prob,
            'expected': dir_type,
            'correct': is_correct
        })
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results) if results else 0
    
    # Print results
    print(f"Results for {dir_type} directory:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    
    # Sample predictions
    print("\nSample predictions:")
    for r in results[:5]:  # Show first 5 results
        print(f"{os.path.basename(r['path'])}: Predicted {r['prediction']} (Confidence: {r['confidence']:.4f}, Raw: {r['raw_score']:.4f}), Expected: {r['expected']}")
    
    # Save results to CSV
    output_file = f"inference_results_{dir_type}.csv"
    with open(output_file, 'w') as f:
        f.write("image,prediction,confidence,raw_score,expected,correct\n")
        for r in results:
            f.write(f"{os.path.basename(r['path'])},{r['prediction']},{r['confidence']:.4f},{r['raw_score']:.4f},{r['expected']},{r['correct']}\n")
    
    print(f"Results saved to {output_file}")
    return accuracy

if __name__ == "__main__":
    main()