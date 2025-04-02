import os
import torch
from PIL import Image
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from networks.trainer import Trainer
from options.train_options import TrainOptions

def fgsm_attack(model, image, target_label, epsilon, device):
    """
    Perform FGSM attack on the model
    
    Args:
        model: The model to attack
        image: Input image tensor (normalized)
        target_label: Target class (0 for fake, 1 for real)
        epsilon: Attack strength
        device: Device to run attack on
    
    Returns:
        perturbed_image: Adversarial example (normalized for model input)
        raw_perturbed_image: Adversarial example in [0,1] range for saving
    """
    # Clone the input and set requires_grad
    image = image.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(image)
    
    # Handle output whether it's a tensor or tuple
    if isinstance(output, tuple):
        output = output[0]
    
    # Target is 1.0 for "real", 0.0 for "fake"
    # Make sure target has the same shape as output
    target = torch.tensor([[target_label]], dtype=torch.float).to(device)
    
    # Calculate loss
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    data_grad = image.grad.data
    
    # Create perturbation
    sign_data_grad = data_grad.sign()
    
    # Denormalize image for perturbation
    # Convert from normalized space to [0,1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    image_denorm = image * std + mean
    
    # Add perturbation
    perturbed_image_raw = image_denorm + epsilon * sign_data_grad
    
    # Clamp to valid image range [0,1]
    perturbed_image_raw = torch.clamp(perturbed_image_raw, 0, 1)
    
    # Re-normalize for model input
    perturbed_image = (perturbed_image_raw - mean) / std
    
    return perturbed_image, perturbed_image_raw

def save_image(tensor, path):
    """Save a tensor as an image"""
    img = transforms.ToPILImage()(tensor.squeeze().cpu())
    img.save(path)

def main():
    # Use the existing option parser to maintain compatibility
    opt = TrainOptions().parse()
    opt.isTrain = False
    
    # Define checkpoint path
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'model_epoch_best.pth')
    real_dir = '/l/users/sarim.hashmi/Thesis/NIPS/Defactify4_Train/Test_set/real'
    fake_dir = '/l/users/sarim.hashmi/Thesis/NIPS/Defactify4_Train/Test_set/fake'
    
    # Create output directory
    output_dir = 'fgsm_results'
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Define epsilon values
    epsilons = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
    
    # SCENARIO 1: Make fake images appear real
    print("\n" + "="*80)
    print("ATTACK SCENARIO 1: Making FAKE images appear REAL")
    print("="*80)
    
    # Create output directory for this attack
    fake_to_real_dir = os.path.join(output_dir, 'fake_to_real')
    os.makedirs(fake_to_real_dir, exist_ok=True)
    
    # Results to track
    fake_to_real_results = []
    
    # Load fake images
    fake_image_paths = glob(os.path.join(fake_dir, '*.jpg'))
    print(f"Found {len(fake_image_paths)} fake images")
    
    # Run attack for each epsilon
    for epsilon in epsilons:
        print(f"\nRunning attack with epsilon = {epsilon}")
        
        # Create directory for this epsilon's results
        if epsilon > 0:
            epsilon_dir = os.path.join(fake_to_real_dir, f'epsilon_{epsilon:.3f}')
            os.makedirs(os.path.join(epsilon_dir, 'successful'), exist_ok=True)
            os.makedirs(os.path.join(epsilon_dir, 'failed'), exist_ok=True)
            os.makedirs(os.path.join(epsilon_dir, 'comparisons'), exist_ok=True)
        
        successful_attacks = 0
        total_images = len(fake_image_paths)
        results = []
        
        # Process each image
        for img_path in tqdm(fake_image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get original prediction
            with torch.no_grad():
                output = trainer.model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                orig_prob = torch.sigmoid(output).item()
                orig_pred = "Real" if orig_prob >= 0.5 else "Fake"
                orig_conf = orig_prob if orig_pred == "Real" else 1 - orig_prob
            
            # Skip attack if already classified as real
            if orig_pred == "Real" and epsilon > 0:
                successful_attacks += 1
                continue
            
            # For epsilon=0, just record original results
            if epsilon == 0:
                if orig_pred == "Real":
                    successful_attacks += 1
                continue
            
            # Perform FGSM attack (target label 1 = "Real")
            perturbed_tensor, perturbed_image = fgsm_attack(
                trainer.model, img_tensor, 1, epsilon, device)
            
            # Evaluate adversarial example
            with torch.no_grad():
                adv_output = trainer.model(perturbed_tensor)
                if isinstance(adv_output, tuple):
                    adv_output = adv_output[0]
                adv_prob = torch.sigmoid(adv_output).item()
                adv_pred = "Real" if adv_prob >= 0.5 else "Fake"
                adv_conf = adv_prob if adv_pred == "Real" else 1 - adv_prob
            
            # Record result
            result = {
                'path': img_path,
                'filename': os.path.basename(img_path),
                'orig_prediction': orig_pred,
                'orig_confidence': orig_conf,
                'orig_raw_score': orig_prob,
                'adv_prediction': adv_pred,
                'adv_confidence': adv_conf,
                'adv_raw_score': adv_prob,
                'success': adv_pred == "Real"
            }
            results.append(result)
            
            # Check if attack was successful
            if adv_pred == "Real":
                successful_attacks += 1
                
                # Save adversarial image
                if epsilon > 0:
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(epsilon_dir, 'successful', img_name)
                    save_image(perturbed_image, save_path)
                    
                    # Create comparison image
                    plt.figure(figsize=(10, 5))
                    
                    # Original image
                    plt.subplot(1, 2, 1)
                    orig_img = transforms.ToPILImage()(img_tensor.squeeze().cpu())
                    plt.imshow(np.array(orig_img))
                    plt.title(f"Original: {orig_conf:.4f} {orig_pred}")
                    plt.axis('off')
                    
                    # Adversarial image
                    plt.subplot(1, 2, 2)
                    adv_img = transforms.ToPILImage()(perturbed_image.squeeze().cpu())
                    plt.imshow(np.array(adv_img))
                    plt.title(f"Adversarial: {adv_conf:.4f} {adv_pred}")
                    plt.axis('off')
                    
                    plt.suptitle(f"ε={epsilon:.3f}", fontsize=16)
                    plt.tight_layout()
                    
                    # Save comparison
                    comp_path = os.path.join(epsilon_dir, 'comparisons', f"compare_{img_name}")
                    plt.savefig(comp_path)
                    plt.close()
            else:
                # Save failed attack
                if epsilon > 0:
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(epsilon_dir, 'failed', img_name)
                    save_image(perturbed_image, save_path)
        
        # Calculate success rate
        success_rate = successful_attacks / total_images
        print(f"Epsilon {epsilon:.3f}: Success rate = {success_rate:.4f} ({successful_attacks}/{total_images})")
        
        # Record success rate for this epsilon
        fake_to_real_results.append(success_rate)
        
        # Save detailed results for this epsilon
        if epsilon > 0 and results:
            # Save to CSV
            csv_path = os.path.join(epsilon_dir, 'results.csv')
            with open(csv_path, 'w') as f:
                f.write("image,orig_pred,orig_conf,orig_raw,adv_pred,adv_conf,adv_raw,success\n")
                for r in results:
                    f.write(f"{r['filename']},{r['orig_prediction']},{r['orig_confidence']:.4f}," +
                           f"{r['orig_raw_score']:.4f},{r['adv_prediction']},{r['adv_confidence']:.4f}," +
                           f"{r['adv_raw_score']:.4f},{r['success']}\n")
    
    # Plot and save success rate graph
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, fake_to_real_results, 'o-', linewidth=2)
    plt.title('FGSM Attack Success Rate (Fake → Real)')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.savefig(os.path.join(fake_to_real_dir, 'success_rate.png'))
    plt.close()
    
    # Save overall results
    with open(os.path.join(fake_to_real_dir, 'summary.csv'), 'w') as f:
        f.write("epsilon,success_rate\n")
        for eps, rate in zip(epsilons, fake_to_real_results):
            f.write(f"{eps:.4f},{rate:.4f}\n")
    
    # SCENARIO 2: Make real images appear fake
    print("\n" + "="*80)
    print("ATTACK SCENARIO 2: Making REAL images appear FAKE")
    print("="*80)
    
    # Create output directory for this attack
    real_to_fake_dir = os.path.join(output_dir, 'real_to_fake')
    os.makedirs(real_to_fake_dir, exist_ok=True)
    
    # Results to track
    real_to_fake_results = []
    
    # Load real images
    real_image_paths = glob(os.path.join(real_dir, '*.jpg'))
    print(f"Found {len(real_image_paths)} real images")
    
    # Run attack for each epsilon
    for epsilon in epsilons:
        print(f"\nRunning attack with epsilon = {epsilon}")
        
        # Create directory for this epsilon's results
        if epsilon > 0:
            epsilon_dir = os.path.join(real_to_fake_dir, f'epsilon_{epsilon:.3f}')
            os.makedirs(os.path.join(epsilon_dir, 'successful'), exist_ok=True)
            os.makedirs(os.path.join(epsilon_dir, 'failed'), exist_ok=True)
            os.makedirs(os.path.join(epsilon_dir, 'comparisons'), exist_ok=True)
        
        successful_attacks = 0
        total_images = len(real_image_paths)
        results = []
        
        # Process each image
        for img_path in tqdm(real_image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get original prediction
            with torch.no_grad():
                output = trainer.model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                orig_prob = torch.sigmoid(output).item()
                orig_pred = "Real" if orig_prob >= 0.5 else "Fake"
                orig_conf = orig_prob if orig_pred == "Real" else 1 - orig_prob
            
            # Skip attack if already classified as fake
            if orig_pred == "Fake" and epsilon > 0:
                successful_attacks += 1
                continue
            
            # For epsilon=0, just record original results
            if epsilon == 0:
                if orig_pred == "Fake":
                    successful_attacks += 1
                continue
            
            # Perform FGSM attack (target label 0 = "Fake")
            perturbed_tensor, perturbed_image = fgsm_attack(
                trainer.model, img_tensor, 0, epsilon, device)
            
            # Evaluate adversarial example
            with torch.no_grad():
                adv_output = trainer.model(perturbed_tensor)
                if isinstance(adv_output, tuple):
                    adv_output = adv_output[0]
                adv_prob = torch.sigmoid(adv_output).item()
                adv_pred = "Real" if adv_prob >= 0.5 else "Fake"
                adv_conf = adv_prob if adv_pred == "Real" else 1 - adv_prob
            
            # Record result
            result = {
                'path': img_path,
                'filename': os.path.basename(img_path),
                'orig_prediction': orig_pred,
                'orig_confidence': orig_conf,
                'orig_raw_score': orig_prob,
                'adv_prediction': adv_pred,
                'adv_confidence': adv_conf,
                'adv_raw_score': adv_prob,
                'success': adv_pred == "Fake"
            }
            results.append(result)
            
            # Check if attack was successful
            if adv_pred == "Fake":
                successful_attacks += 1
                
                # Save adversarial image
                if epsilon > 0:
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(epsilon_dir, 'successful', img_name)
                    save_image(perturbed_image, save_path)
                    
                    # Create comparison image
                    plt.figure(figsize=(10, 5))
                    
                    # Original image
                    plt.subplot(1, 2, 1)
                    orig_img = transforms.ToPILImage()(img_tensor.squeeze().cpu())
                    plt.imshow(np.array(orig_img))
                    plt.title(f"Original: {orig_conf:.4f} {orig_pred}")
                    plt.axis('off')
                    
                    # Adversarial image
                    plt.subplot(1, 2, 2)
                    adv_img = transforms.ToPILImage()(perturbed_image.squeeze().cpu())
                    plt.imshow(np.array(adv_img))
                    plt.title(f"Adversarial: {adv_conf:.4f} {adv_pred}")
                    plt.axis('off')
                    
                    plt.suptitle(f"ε={epsilon:.3f}", fontsize=16)
                    plt.tight_layout()
                    
                    # Save comparison
                    comp_path = os.path.join(epsilon_dir, 'comparisons', f"compare_{img_name}")
                    plt.savefig(comp_path)
                    plt.close()
            else:
                # Save failed attack
                if epsilon > 0:
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(epsilon_dir, 'failed', img_name)
                    save_image(perturbed_image, save_path)
        
        # Calculate success rate
        success_rate = successful_attacks / total_images
        print(f"Epsilon {epsilon:.3f}: Success rate = {success_rate:.4f} ({successful_attacks}/{total_images})")
        
        # Record success rate for this epsilon
        real_to_fake_results.append(success_rate)
        
        # Save detailed results for this epsilon
        if epsilon > 0 and results:
            # Save to CSV
            csv_path = os.path.join(epsilon_dir, 'results.csv')
            with open(csv_path, 'w') as f:
                f.write("image,orig_pred,orig_conf,orig_raw,adv_pred,adv_conf,adv_raw,success\n")
                for r in results:
                    f.write(f"{r['filename']},{r['orig_prediction']},{r['orig_confidence']:.4f}," +
                           f"{r['orig_raw_score']:.4f},{r['adv_prediction']},{r['adv_confidence']:.4f}," +
                           f"{r['adv_raw_score']:.4f},{r['success']}\n")
    
    # Plot and save success rate graph
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, real_to_fake_results, 'o-', linewidth=2)
    plt.title('FGSM Attack Success Rate (Real → Fake)')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.savefig(os.path.join(real_to_fake_dir, 'success_rate.png'))
    plt.close()
    
    # Save overall results
    with open(os.path.join(real_to_fake_dir, 'summary.csv'), 'w') as f:
        f.write("epsilon,success_rate\n")
        for eps, rate in zip(epsilons, real_to_fake_results):
            f.write(f"{eps:.4f},{rate:.4f}\n")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    plt.plot(epsilons, fake_to_real_results, 'o-', linewidth=2, label='Fake → Real')
    plt.plot(epsilons, real_to_fake_results, 's-', linewidth=2, label='Real → Fake')
    plt.title('FGSM Attack Success Rate Comparison')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    plt.close()
    
    print(f"\nAttack complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()