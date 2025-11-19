import torch
import pickle
import os
from typing import Dict, List, Any
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


class StepLogger:
    """Simple step-logger for OnPolicyRunner"""

    def __init__(self, log_step_flag: bool = False):
        self.rewards = []
        self.dones = []
        self.observations = []
        self.actions = []
        self.infos = []
        self.log_step_flag = log_step_flag
        self.start_step = None

    def log_step(self, obs: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, dones: torch.Tensor,
                 infos: List[Dict[str, Any]], timestep: int):
        """Log a single step's data"""
        if self.log_step_flag is False:
            return
        
        if not self.start_step:
            self.start_step = timestep
        
        self.rewards.append(rewards.cpu())
        self.dones.append(dones.cpu())
        self.observations.append(obs.cpu())
        self.actions.append(actions.cpu())
        self.infos.append(infos.copy())

    def save(self, filepath: str):
        """Save all logged data to pickle file"""
        if self.log_step_flag is False:
            return
        if len(self.rewards) == 0:
            print("No data to save")
            return

        print(f"Saving {len(self.rewards)} steps to {filepath}")

        # Convert lists to tensors
        data = {
            'start_step': self.start_step,
            'rewards': torch.stack(self.rewards),
            'dones': torch.stack(self.dones),
            'observations': torch.stack(self.observations),
            'actions': torch.stack(self.actions),
            'infos': self.infos,
            'num_steps': len(self.rewards),
            'timestamp': datetime.now().isoformat()
        }

        # Save to pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Saved to {filepath}")
        self.clear()

    def clear(self):
        """Reset for next"""
        self.start_step = None
        self.rewards.clear()
        self.dones.clear()
        self.observations.clear()
        self.actions.clear()
        self.infos.clear()

class AsyncSaver:
    def __init__(self, quality=85, max_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.quality = quality
        print(f"Initialized AsyncSaver with {max_workers} workers")
    
    def save_frame(self, env, frameidx, iteration, base_dir, env_idx=0):
        """Save all camera images for one frame from specific environment"""
        camera_names = ['bird', 'front', 'side', 'hand']
        camera_keys = ['camera_bird', 'camera_ext1', 'camera_ext2', 'camera']
        
        for camera_name, camera_key in zip(camera_names, camera_keys):
            try:
                # shape [num_envs, H, W, C]
                tensor = env.unwrapped.scene[camera_key].data.output["rgb"] / 255.0
                
                # specific environment: [H, W, C]
                env_tensor = tensor[env_idx]
                timestep = iteration * 24 + frameidx
                filepath = os.path.join(base_dir, "frames", f"env_{env_idx:03d}", camera_name, f"rgb_out{timestep:08d}.jpg")
                self.executor.submit(self._save_single, env_tensor.clone(), filepath)
                
            except Exception as e:
                print(f"Error accessing camera {camera_name}: {e}")
                return False
        return True
    
    def save_all_envs(self, env, frameidx, iteration, base_dir):
        """Save images for all environments"""
        try:
            # Get number of environments from tensor shape
            sample_tensor = env.unwrapped.scene["camera_bird"].data.output["rgb"] / 255.0
            num_envs = sample_tensor.shape[0]
            
            for env_idx in range(num_envs):
                self.save_frame(env, frameidx, iteration, base_dir, env_idx)
                
        except Exception as e:
            print(f"Error in save_all_envs: {e}")
    
    def _save_single(self, tensor, filepath):
        """Save single image (runs in background thread)"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert tensor to numpy: [H, W, C] format
            img_np = (tensor.detach().cpu().numpy() * 255).astype('uint8')
            
            # Save as JPEG
            Image.fromarray(img_np).save(filepath, 'JPEG', quality=self.quality, optimize=True)
            
        except Exception as e:
            print(f"Failed to save {filepath}: {e}")
    
    def shutdown(self):
        """Call this when training ends"""
        self.executor.shutdown(wait=True)