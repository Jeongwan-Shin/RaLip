import os

import numpy as np
import torch
from torch.utils.data import Dataset

MARS_to_STD = {
    0: 0,    # pelvis -> pelvis
    18: 1,   #  thorax -> neck
    3: 2,    #  head -> head
    4: 3,    #  left_shoulder -> left_shoulder
    5: 4,    #  left_elbow -> left_elbow
    6: 5,    #  left_wrist -> left_wrist
    7: 6,    #  right_shouler -> right_shouler
    8: 7,    #  right_elbow -> right_elbow
    9: 8,    #  right_wrist -> right_wrist
    10: 9,   #  left_hip -> left_hip
    11: 10,  #  left_knee -> left_knee
    12: 11,  #  left_ankle -> left_ankle
    14: 12,  #  right_hip -> right_hip
    15: 13,  #  right_knee -> right_knee
    16: 14,  #  right_ankle -> right_ankle
}

# NOTE: placeholder (kept for backward-compat). Define the correct mapping if needed.
MiliPoint_to_STD = {}

mRI_to_STD = {
    'pelvis': 0,    # left/right_hip mean -> pelvis (11+12 idx /2)
    'neck': 1,    # left/right_shoulder mean -> neck (5+6 idx /2)
    0: 2,    #  nose -> head
    6: 3,    #  left_shoulder -> left_shoulder
    8: 4,    #  left_elbow -> left_elbow
    10: 5,    #  left_wrist -> left_wrist
    5: 6,    #  right_shouler -> right_shouler
    7: 7,    #  right_elbow -> right_elbow
    9: 8,    #  right_wrist -> right_wrist
    12: 9,   #  left_hip -> left_hip
    14: 10,  #  left_knee -> left_knee
    16: 11,  #  left_ankle -> left_ankle
    11: 12,  #  right_hip -> right_hip
    13: 13,  #  right_knee -> right_knee
    15: 14,  #  right_ankle -> right_ankle
}

mmFI_to_STD = {
    0: 0,    # pelvis -> pelvis
    8: 1,    # neck -> neck
    9: 2,    # nose -> head
    11: 3,    # left_shoulder -> left_shoulder
    12: 4,    # left_elbow -> left_elbow
    13: 5,    # left_wrist -> left_wrist
    14: 6,    # right_shouler -> right_shouler
    15: 7,    # right_elbow -> right_elbow
    16: 8,    # right_wrist -> right_wrist
    4: 9,   # left_hip -> left_hip
    5: 10,  # left_knee -> left_knee
    6: 11,  # left_ankle -> left_ankle
    1: 12,  # right_hip -> right_hip
    2: 13,  # right_knee -> right_knee
    3: 14,  # right_ankle -> right_ankle
}

def train_test_cross_split(data_dir, train_list, test_list):

    train_files = []
    test_files = []

    # train_data_list
    for subject in train_list:
        subject_path = os.path.join(data_dir, subject)

        npz_files = sorted([
            os.path.join(subject_path, f)
            for f in os.listdir(subject_path)
            if f.endswith(".npz")
        ])
        train_files.extend(npz_files)

    # test_data_list
    for subject in test_list:
        subject_path = os.path.join(data_dir, subject)

        npz_files = sorted([
            os.path.join(subject_path, f)
            for f in os.listdir(subject_path)
            if f.endswith(".npz")
        ])
        test_files.extend(npz_files)

    return train_files, test_files

def jitter(points: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    points: (N, 5) numpy array [x, y, z, doppler, intensity]
    sigma: Gaussian noise std-dev for xyz (meters). default 2cm.
    """
    noisy_points = points.copy()
    n = points.shape[0]
    noise = np.random.normal(loc=0.0, scale=sigma, size=(n, 3))
    noisy_points[:, :3] += noise
    return noisy_points


def apply_feature_norm(points: np.ndarray, mode: str) -> np.ndarray:
    """
    points: (N,5) numpy array [x,y,z,doppler,intensity]
    mode:
      - 'none'
      - 'per_sample_zscore': z-score normalize xyz per sample (after zero-pad removal)
    """
    mode = str(mode or "none")
    if mode == "none":
        return points
    if mode == "per_sample_zscore":
        xyz = points[:, :3]
        mu = xyz.mean(axis=0, keepdims=True)
        std = xyz.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        points = points.copy()
        points[:, :3] = (xyz - mu) / std
        return points
    raise ValueError(f"Unknown feature_norm mode: {mode}")

class mRI_Dataset(Dataset):
    def __init__(self, file_list, transform=False, train=True, *, feature_norm: str = "none"):

        self.file_list = file_list

        self.transform = transform
        self.train = train
        self.feature_norm = feature_norm

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True, mmap_mode=None)

        feature = data['feature'].copy().reshape(-1, 5) # 14 x 14 x 5
        label = data['label'].copy()
        label = org_label_to_STD_label(label[[0, 2, 1], :], "mRI") # org data has changed y,z

        feature = remove_zero_padded_points(feature)
        if feature.shape[0] == 0:
            feature = np.zeros((1, 5))
        feature = apply_feature_norm(feature, self.feature_norm)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def org_label_to_STD_label(keypoints, dataset):
    
    if dataset == "MARS":
        dict = MARS_to_STD

    elif dataset == "mRI":
        dict = mRI_to_STD

    elif dataset == "mmFI":
        dict = mmFI_to_STD

    elif dataset == "MiliPoint":
        dict = MiliPoint_to_STD

    STD_keypoints = np.zeros((3, 15))

    for org_idx, STD_idx in dict.items():

        if org_idx == 'pelvis':
            pass
        elif org_idx == 'neck':
            pass

        else:
            STD_keypoints[:, STD_idx] = keypoints[:, org_idx]

    try:
        STD_idx_p = dict["pelvis"]

        STD_keypoints[:,STD_idx_p] = (STD_keypoints[:, 9] + STD_keypoints[:, 12])/2 # 9,12 idx (left, right hip)

        STD_idx_n = dict["neck"]

        STD_keypoints[:,STD_idx_n] = (STD_keypoints[:, 3] + STD_keypoints[:, 6])/2 # 3,6 idx (left, right shoulder)

    except:
        pass
    
    return STD_keypoints.reshape(3*15)

def mmBody_train_test_path(mmBody_dir):

    train_root = os.path.join(mmBody_dir, "train")
    test_root = os.path.join(mmBody_dir, "test")

    def get_all_npz_files(root_dir):
        file_list = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith('.npz'):
                    file_list.append(os.path.join(dirpath, fname))
        return file_list

    train_data_list = get_all_npz_files(train_root)
    test_data_list = get_all_npz_files(test_root)

    return train_data_list, test_data_list
    
def remove_zero_padded_points(pc):
    zero_mask = np.all(pc == 0, axis=1)  # True: 제로 패딩 포인트
    filtered_pc = pc[~zero_mask]

    return filtered_pc

class MARS_Dataset(Dataset):
    def __init__(self, folder_path, transform=False, train=True, *, feature_norm: str = "none"):

        self.folder_path = folder_path
        self.file_list = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.npz')
        ])

        self.transform = transform
        self.train = train
        self.feature_norm = feature_norm

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True, mmap_mode=None)

        feature = data['feature'].copy().reshape(-1, 5) # 8 x 8 x 5 -> 64 x 5
        label = data['label'].copy()

        label = org_label_to_STD_label(label.reshape(3, 19), "MARS")


        feature = remove_zero_padded_points(feature)
        if feature.shape[0] == 0:
            feature = np.zeros((1, 5))
        feature = apply_feature_norm(feature, self.feature_norm)
        
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class mmFI_Dataset(Dataset):
    def __init__(self, file_list, transform=False, train=True, *, feature_norm: str = "none"):
        self.file_list = file_list
        self.transform = transform
        self.train = train   
        self.feature_norm = feature_norm

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True, mmap_mode=None)

        feature = data['feature'].copy() # K x 5
        label = data['label'].copy()

        feature = feature[:, [1,0,2,3,4]] # org data has changed x,y

        label[:,1] = label[:,1] * -1   # org label has inverted about the xy plane (person is upside down)
        label = org_label_to_STD_label(np.transpose(label[:, [0, 2, 1]]), "mmFI")
        
        feature = remove_zero_padded_points(feature)
        if feature.shape[0] == 0:
            feature = np.zeros((1, 5))
        feature = apply_feature_norm(feature, self.feature_norm)
        if self.transform and self.train:
            feature = jitter(feature) # aug 1: jitter


        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class mmBody_Dataset(Dataset):
    def __init__(self, file_list, transform=False, train=True, *, feature_norm: str = "none"):
        self.file_list = file_list
        self.transform = transform
        self.train = train
        self.feature_norm = feature_norm
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True, mmap_mode=None)

        feature = data['feature'].copy()
        label = data['label']

        feature = feature[:,:5]
        label = label

        feature = remove_zero_padded_points(feature)
        if feature.shape[0] == 0:
            feature = np.zeros((1, 5))
        feature = apply_feature_norm(feature, self.feature_norm)
        if self.transform and self.train:
            feature = jitter(feature)
            
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)