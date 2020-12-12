import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
import multiprocessing

import augmix


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        # os.mkdir(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,
                 pre_image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size # for translation
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False




    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = fast_random_crop(obses, self.image_size)
        next_obses = fast_random_crop(next_obses, self.image_size)
        pos = fast_random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def sample_rad(self,aug_funcs):

        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler


        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug:
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_augmix(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        clean_obses = obses.copy()
        next_obses = self.next_obses[idxs]
        clean_next_obses = next_obses.copy()

        clean_obses = torch.as_tensor(clean_obses, device=self.device).float()
        clean_next_obses = torch.as_tensor(clean_next_obses, device=self.device).float()


        obses = obses / 255.
        next_obses = next_obses / 255.
        clean_obses = clean_obses / 255.
        clean_next_obses = clean_next_obses / 255.

        ### augmix ####

        N_PROCS = 0
        if N_PROCS ==0:
            obses = [augmix.augment_and_mix(obses[i], seed=i)
                                    for i in range(len(obses))]
            next_obses = [augmix.augment_and_mix(next_obses[i], seed=i)
                                for i in range(len(next_obses))]

        else: # running in parallel
            with multiprocessing.Pool(processes=N_PROCS) as pool:
                args_list = [(obses[i], i) for i in range(len(obses))]
                obses = pool.starmap(augmix.augment_and_mix2, args_list)
                args_list = [(next_obses[i], i) for i in range(len(next_obses))]
                next_obses = pool.starmap(augmix.augment_and_mix2, args_list)

        ## stacking ##
        obses = [torch.cat(images, 0) for images in obses]
        next_obses = [torch.cat(images, 0) for images in next_obses]

        obses = torch.stack(obses).float().to(self.device)
        next_obses = torch.stack(next_obses).float().to(self.device)

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, clean_obses, actions, rewards, next_obses, clean_next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1:h1 + h, w1:w1 + w] = image
    return outs


class AugmixReplayBuffer(Dataset):
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,
                 pre_image_size=84, transform=None, aug_cache = 4, aug_depth=1 , aug_width =1, aug_alpha = 1.0 ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size # for translation
        self.transform = transform
        self.aug_cache = aug_cache
        self.aug_depth = aug_depth
        self.aug_width = aug_width
        self.aug_alpha = aug_alpha
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.aug_obses = [np.empty((capacity, *obs_shape), dtype=obs_dtype) for _ in range(self.aug_cache)]
        self.aug_next_obses = [np.empty((capacity, *obs_shape), dtype=obs_dtype) for _ in range(self.aug_cache)]

        self.idx = 0
        self.last_save = 0
        self.full = False




    def add(self, obs, action, reward, next_obs, done):
        # import ipdb; ipdb.set_trace()
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        def get_augmented_obs(obs):
            severity = 3

            # apply random augmentation here
            from augmix import augmentations, apply_op
            obs = obs /255.

            depth = self.aug_depth
            c, h, w = obs.shape
            assert c == 9
            obs_images = obs.reshape((-1,3,h,w))

            # do we need to apply same operation to all 3 conseqence images?

            aug_obs_images = []
            for image in obs_images:
                image_aug = image.transpose((1,2,0)).copy() # h, w , c
                d = depth if depth > 0 else np.random.randint(1, 4)
                op = np.random.choice(augmentations, size=d, replace=True)
                for i in range(d):
                    image_aug = apply_op(image_aug, op[i], severity)
                aug_obs_images.append(image_aug.transpose((2,0,1))) # c h w

            aug_obs_images = np.stack(aug_obs_images, axis = 0)
            aug_obs = aug_obs_images.reshape((9,h,w))

            aug_obs = np.clip(aug_obs*255., 0, 255).astype(np.uint8)
            return aug_obs

        for c_i in range(self.aug_cache):
            aug_obs = get_augmented_obs(obs)
            np.copyto(self.aug_obses[c_i][self.idx], aug_obs)
            aug_next_obs = get_augmented_obs(next_obs)
            np.copyto(self.aug_next_obses[c_i][self.idx], aug_next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


        # check 255 ranges of original obs



    def sample_augmix(self):

        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler


        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        aug_obses = [ self.aug_obses[c_i][idxs] for c_i in range(self.aug_cache)]
        aug_next_obses = [ self.aug_next_obses[c_i][idxs] for c_i in range(self.aug_cache)]

        # import ipdb; ipdb.set_trace()

        # augmix mixing part
        def get_mixed_obs(obses, aug_obses):
            # this part is slow...
            # 1) try batch wighted sum emthods numpy
            # 2) maye also use cuda()
            # * why was 0 appearing in augmented obs? take a look
            from augmix import normalize_chw

            width = self.aug_width
            alpha = self.aug_alpha

            c, h, w = obses[0].shape

            t00= time.time()

            obs_images = obses.reshape((-1,3,h,w))
            aug_obs_images = [ aug_obses1.reshape((-1,3,h,w)) for aug_obses1 in aug_obses]

            mixed_images = []

            t01= time.time()

            for i in range(obs_images.shape[0]):
                ws = np.float32(np.random.dirichlet([alpha] * width))
                m = np.float32(np.random.beta(alpha, alpha))
                mix = np.zeros((3, h, w)).astype(float)

                cache_indices = np.random.choice(self.aug_cache, size= width, replace=False)

                for w_i in range(width):
                    image_aug = aug_obs_images[cache_indices[w_i]][i]
                    mix = np.add(mix, ws[w_i] * normalize_chw(image_aug), casting='no')

                image = obs_images[i]
                mixed = (1-m) * normalize_chw(image) + m * mix

                mixed_images.append(mixed)
            t02= time.time()
            mixed_images = np.stack(mixed_images, axis=0)
            mixed_images = mixed_images.reshape((-1,9,h,w))
            t03= time.time()

            # print( t01-t00,t02-t01, t03-t02)

            return mixed_images

        mixed_obses = get_mixed_obs(obses, aug_obses)
        mixed_next_obses = get_mixed_obs(next_obses, aug_next_obses)

        clean_obses = torch.as_tensor(obses, device=self.device).float()
        clean_next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses = torch.as_tensor(mixed_obses, device=self.device).float()
        next_obses = torch.as_tensor(mixed_next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        clean_obses = clean_obses / 255.
        next_obses = next_obses / 255.
        clean_next_obses = clean_next_obses / 255.

        return obses, clean_obses, actions, rewards, next_obses,clean_next_obses, not_dones
