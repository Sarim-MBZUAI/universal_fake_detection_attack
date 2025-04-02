import os
import time
import random
import numpy as np
from glob import glob
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from validate import validate
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


class DeepfakeDataset(Dataset):
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.is_train = is_train
        
        # Use the correct option names from TrainOptions
        self.transform = transforms.Compose([
            transforms.Resize((opt.cropSize, opt.cropSize)),
            transforms.RandomHorizontalFlip() if is_train and not opt.no_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define data directory paths
        self.base_dir = os.path.join('/l/users/sarim.hashmi/Thesis/NIPS/Defactify4_Train/Train_set')
        
        # Load all image paths
        self.real_images = glob(os.path.join(self.base_dir, 'real', '*.jpg'))
        self.fake_images = glob(os.path.join(self.base_dir, 'fake', '*.jpg'))
        
        # Split for train/val if needed
        if is_train:
            self.real_images = self.real_images[:int(len(self.real_images) * 0.8)]
            self.fake_images = self.fake_images[:int(len(self.fake_images) * 0.8)]
        else:
            self.real_images = self.real_images[int(len(self.real_images) * 0.8):]
            self.fake_images = self.fake_images[int(len(self.fake_images) * 0.8):]
        
        # Combine and create labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = [1] * len(self.real_images) + [0] * len(self.fake_images)
        
        # Shuffle data
        if not opt.serial_batches:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
        
        print(f"Loaded {len(self.image_paths)} images: {len(self.real_images)} real, {len(self.fake_images)} fake")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # For validation, only return image and label
        if not self.is_train:
            return image, torch.tensor(label, dtype=torch.float)
        
        # For training, return with path for model.set_input()
        return [image, torch.tensor(label, dtype=torch.float), img_path]


def create_dataloader(opt, is_train=True):
    dataset = DeepfakeDataset(opt, is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches and is_train,
        num_workers=opt.num_threads
    )
    return dataloader


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_flip = True
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    
    # Handle blur_sig properly based on its type
    if isinstance(val_opt.blur_sig, list) and len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    elif isinstance(val_opt.blur_sig, str) and ',' in val_opt.blur_sig:
        b_sig = [float(s) for s in val_opt.blur_sig.split(',')]
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
        
    # Handle jpg_qual properly
    if isinstance(val_opt.jpg_qual, list) and len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    elif isinstance(val_opt.jpg_qual, str) and ',' in val_opt.jpg_qual:
        j_qual = [int(q) for q in val_opt.jpg_qual.split(',')]
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    # Set random seed for reproducibility
    seed = getattr(opt, 'seed', 42)  # Use 42 as default if seed is not specified
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt, is_train=True)
    val_loader = create_dataloader(val_opt, is_train=False)

    # Create directories if they don't exist
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))
    
    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps))

            # Avoid using opt.save_checkpoint_steps which may not exist
            if model.total_steps in [10, 30, 50, 100, 1000, 5000, 10000] and False:
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks('model_epoch_best.pth')
            model.save_networks('model_epoch_%s.pth' % epoch)

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        val_writer.add_scalar('real_accuracy', r_acc, model.total_steps)
        val_writer.add_scalar('fake_accuracy', f_acc, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}; real_acc: {}; fake_acc: {}".format(
            epoch, acc, ap, r_acc, f_acc))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()