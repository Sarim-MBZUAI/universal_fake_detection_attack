import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
from models import get_model
from PIL import Image 
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
from copy import deepcopy

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        
 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def get_image_list(directory):
    """Get all image files from a directory"""
    image_list = []
    valid_extensions = ["png", "jpg", "jpeg", "JPEG", "bmp"]
    
    for filename in os.listdir(directory):
        extension = filename.split('.')[-1]
        if extension in valid_extensions:
            image_list.append(os.path.join(directory, filename))
    
    return image_list


class SimpleDataset(Dataset):
    def __init__(self, real_path, fake_path, max_sample, arch, jpeg_quality=None, gaussian_sigma=None):
        """
        A simplified dataset class that works directly with real/fake directories
        
        Parameters:
        real_path (str): Directory containing real images
        fake_path (str): Directory containing fake images
        max_sample (int): Maximum number of samples to use from each class
        arch (str): Architecture name for normalization
        jpeg_quality (int, optional): JPEG compression quality for robustness testing
        gaussian_sigma (int, optional): Gaussian blur sigma for robustness testing
        """
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # Get image lists
        real_list = get_image_list(real_path)
        fake_list = get_image_list(fake_path)
        
        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                available = min(len(real_list), len(fake_list))
                print(f"Warning: Not enough images, max_sample falling to {available}")
                max_sample = available
                
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[:max_sample]
            fake_list = fake_list[:max_sample]
        
        print(f"Using {len(real_list)} real images and {len(fake_list)} fake images")
        
        self.total_list = real_list + fake_list
        
        # Create labels dictionary
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0  # 0 for real
        for i in fake_list:
            self.labels_dict[i] = 1  # 1 for fake
            
        # Set up transformations
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--fake_path', type=str, required=True, help='Directory containing fake images')
    parser.add_argument('--max_sample', type=int, default=None, help='Only check this number of images for both fake/real')
    
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14', help='Model architecture')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth', help='Path to model weights')
    
    parser.add_argument('--result_folder', type=str, default='result', help='Folder to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    parser.add_argument('--jpeg_quality', type=int, default=None, help="JPEG quality for robustness testing")
    parser.add_argument('--gaussian_sigma', type=float, default=None, help="Gaussian blur sigma for robustness testing")
    
    opt = parser.parse_args()
    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)
    
    # Load model
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print("Model loaded from", opt.ckpt)
    model.eval()
    model.cuda()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Create dataset and dataloader
    dataset = SimpleDataset(
        opt.real_path,
        opt.fake_path,
        opt.max_sample,
        opt.arch,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Run validation
    ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
    
    # Save results
    result_file = os.path.join(opt.result_folder, 'results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {opt.real_path} (real) and {opt.fake_path} (fake)\n")
        f.write(f"Model: {opt.arch}\n")
        f.write(f"Checkpoint: {opt.ckpt}\n\n")
        
        f.write(f"Average Precision: {ap*100:.2f}%\n\n")
        
        f.write("With threshold = 0.5:\n")
        f.write(f"  Real accuracy: {r_acc0*100:.2f}%\n")
        f.write(f"  Fake accuracy: {f_acc0*100:.2f}%\n")
        f.write(f"  Overall accuracy: {acc0*100:.2f}%\n\n")
        
        f.write(f"Best threshold: {best_thres:.4f}\n")
        f.write("With best threshold:\n")
        f.write(f"  Real accuracy: {r_acc1*100:.2f}%\n")
        f.write(f"  Fake accuracy: {f_acc1*100:.2f}%\n")
        f.write(f"  Overall accuracy: {acc1*100:.2f}%\n")
    
    print(f"Results saved to {result_file}")
    
    # Also save individual metrics to separate files for compatibility
    with open(os.path.join(opt.result_folder, 'ap.txt'), 'w') as f:
        f.write(f"AP: {ap*100:.2f}\n")
    
    with open(os.path.join(opt.result_folder, 'acc0.txt'), 'w') as f:
        f.write(f"Real: {r_acc0*100:.2f}  Fake: {f_acc0*100:.2f}  Overall: {acc0*100:.2f}\n")
    
    with open(os.path.join(opt.result_folder, 'best_threshold.txt'), 'w') as f:
        f.write(f"Best threshold: {best_thres:.4f}\n")
        f.write(f"Real: {r_acc1*100:.2f}  Fake: {f_acc1*100:.2f}  Overall: {acc1*100:.2f}\n")