import os
import numpy as np
import cv2
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms

def generate_heatmap(shape, joint, sigma=1):
    """
    Generate Gaussian heatmap.
    :param shape: Tuple (H, W), the shape of the heatmap
    :param joint: Tuple (x, y), coordinates of the joint
    :param sigma: Standard deviation of the Gaussian distribution
    :return: Heatmap
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    if joint[0] < 0 or joint[1] < 0:  # Invisible joint
        return heatmap
    
    x, y = joint
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap = np.clip(heatmap, 0, 1)  # Normalize to [0, 1]
    return heatmap

def get_transform():
    """Define data preprocessing"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_datasets(image_dir, joints_path, sigma, transform, seed=42):
    """Load and split the dataset"""
    dataset = LSPDataset(
        image_dir=image_dir,
        joints_path=joints_path,
        sigma=sigma,
        transform=transform
    )

    torch.manual_seed(seed)

    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8):
    """Create DataLoader"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class LSPDataset(Dataset):
    def __init__(self, image_dir, joints_path, output_size=(368, 368), heatmap_size=(45, 45), sigma=1, transform=None):
        """
        Initialize the LSP dataset.
        :param image_dir: Path to the image folder
        :param joints_path: Path to the joints.mat file
        :param output_size: Output image size (default 368x368)
        :param heatmap_size: Size of the output heatmap
        :param sigma: Standard deviation of the Gaussian distribution for the heatmap
        :param transform: Data augmentation (e.g., normalization, etc.)
        """
        self.image_dir = image_dir
        self.joints = sio.loadmat(joints_path)['joints']  # Shape of joints: (14, 3, 10000)
        self.output_size = output_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # Original size (H, W)

        # Load joint data
        joints = self.joints[:, :, idx]  # (14, 3), joint coordinates and visibility
        visibility = joints[:, 2]  # Extract visibility flags (14,)
        
        # Synchronously process images and keypoints
        image, adjusted_joints = self._resize_and_adjust_keypoints(image, joints, original_size)

        # Generate heatmap
        heatmaps = self._generate_heatmaps(adjusted_joints)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Return image and heatmap labels
        return image, torch.from_numpy(heatmaps), torch.from_numpy(visibility).float()

    def _resize_and_adjust_keypoints(self, image, joints, original_size):
        """
        Synchronously process images and keypoints: scale the image proportionally and pad it, while adjusting the keypoint coordinates.
        :param image: Original image.
        :param joints: Original joint data (14, 3).
        :param original_size: Original image size (H, W).
        :return: Processed image and adjusted keypoints.
        """
        H, W = original_size
        H_target, W_target = self.output_size

        # 1. Calculate the proportional scaling factor
        scale = min(W_target / W, H_target / H)
        new_W = int(W * scale)
        new_H = int(H * scale)

        # 2. Scale the image proportionally
        resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        # 3. Calculate padding values
        pad_w = (W_target - new_W) // 2
        pad_h = (H_target - new_H) // 2

        # 4. Pad the image
        padded_image = cv2.copyMakeBorder(
            resized_image, pad_h, H_target - new_H - pad_h, pad_w, W_target - new_W - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # 5. Adjust joint coordinates
        adjusted_joints = joints.copy()
        for i, (x, y, v) in enumerate(joints):
            if v > 0:  # Adjust visible joints only
                new_x = x * scale + pad_w
                new_y = y * scale + pad_h
                adjusted_joints[i, 0] = new_x
                adjusted_joints[i, 1] = new_y

        return padded_image, adjusted_joints

    def _generate_heatmaps(self, joints):
        """
        Generate heatmaps for all joints.
        :param joints: Joint coordinates (14, 3)
        :return: Heatmap tensor (14, heatmap_size, heatmap_size)
        """
        heatmaps = np.zeros((joints.shape[0], self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)

        # Scale joint coordinates to heatmap size
        scale = self.heatmap_size[0] / self.output_size[0]
        for i, joint in enumerate(joints):
            if joint[2] > 0:  # Visibility check
                scaled_joint = joint[:2] * scale
                heatmaps[i] = generate_heatmap((self.heatmap_size[0], self.heatmap_size[1]), scaled_joint, self.sigma)

        return heatmaps
    