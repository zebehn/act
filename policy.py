import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    @staticmethod
    def _normalize_image(image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return normalize(image)

    def _forward_model(self, qpos, image, actions=None, is_pad=None, return_aux=False, posterior_decode_mode='sample'):
        env_state = None
        image = self._normalize_image(image)
        return self.model(
            qpos,
            image,
            env_state,
            actions,
            is_pad,
            return_aux=return_aux,
            posterior_decode_mode=posterior_decode_mode,
        )

    def __call__(self, qpos, image, actions=None, is_pad=None):
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self._forward_model(qpos, image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self._forward_model(qpos, image)  # no action, sample from prior
            return a_hat

    def get_action_distribution(self, qpos, image):
        a_hat, _, (_, _), aux = self._forward_model(qpos, image, return_aux=True)
        dist = Normal(a_hat, aux['action_log_std'].exp())
        return {
            'mean': a_hat,
            'log_std': aux['action_log_std'],
            'dist': dist,
            'state_value': aux['state_value'].squeeze(-1),
            'hidden_states': aux['hidden_states'],
        }

    def sample_action(self, qpos, image, query_index=0, deterministic=False):
        policy_output = self.get_action_distribution(qpos, image)
        dist = policy_output['dist'][:, query_index]
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return {
            'action': action,
            'mean': dist.mean,
            'std': dist.stddev,
            'log_prob': log_prob,
            'entropy': entropy,
            'state_value': policy_output['state_value'],
            'query_index': query_index,
        }

    def evaluate_action(self, qpos, image, action, query_index=0):
        policy_output = self.get_action_distribution(qpos, image)
        dist = policy_output['dist'][:, query_index]
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return {
            'log_prob': log_prob,
            'entropy': entropy,
            'state_value': policy_output['state_value'],
            'mean': dist.mean,
            'std': dist.stddev,
            'query_index': query_index,
        }

    def score_action_chunk(self, qpos, image, actions, is_pad, posterior_decode_mode='mean', kl_coef=1.0):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        a_hat, _, (mu, logvar), aux = self._forward_model(
            qpos,
            image,
            actions,
            is_pad,
            return_aux=True,
            posterior_decode_mode=posterior_decode_mode,
        )
        dist = Normal(a_hat, aux['action_log_std'].exp())
        valid_mask = (~is_pad).unsqueeze(-1).float()
        log_prob_terms = dist.log_prob(actions) * valid_mask
        entropy_terms = dist.entropy() * valid_mask
        total_log_prob = log_prob_terms.sum(dim=(1, 2))
        total_entropy = entropy_terms.sum(dim=(1, 2))
        token_count = valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
        total_kld, _, _ = kl_divergence(mu, logvar)
        kl_value = total_kld.view(1) if total_kld.ndim == 0 else total_kld
        score = total_log_prob - kl_coef * kl_value
        return {
            'score': score,
            'mean_score': score / token_count,
            'log_prob': total_log_prob,
            'mean_log_prob': total_log_prob / token_count,
            'entropy': total_entropy,
            'mean_entropy': total_entropy / token_count,
            'kl': kl_value,
            'token_count': token_count.to(dtype=torch.int64),
            'posterior_decode_mode': posterior_decode_mode,
        }

    def load_state_dict(self, state_dict, strict=True):
        missing_allowed = {'model.action_log_std', 'model.value_head.weight', 'model.value_head.bias'}
        incompatible = super().load_state_dict(state_dict, strict=False)
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        missing_disallowed = missing - missing_allowed
        if strict and (unexpected or missing_disallowed):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: missing={sorted(missing_disallowed)}, unexpected={sorted(unexpected)}"
            )
        return incompatible

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
