import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch import autograd

from agent import CUDAAgent

from .base_student import BaseStudent


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def flatten(_list):
    return [item for sublist in _list for item in sublist]


# pylint: disable=arguments-differ
class BCIRMStudent(BaseStudent, CUDAAgent):
    def __init__(
        self,
        env,
        trajs_paths,
        model_path,
        num_training_envs,
        teacher,
        causal_features_encoder,
        policy_network,
        buffer,
        adam_alpha,
        config
    ):
        super(BCIRMStudent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
            model_path=model_path,
            teacher=teacher,
            buffer=buffer,
        )

        self.num_training_envs = num_training_envs

        self.causal_features_encoder = causal_features_encoder.to(self.device)
        self.policy_network = policy_network.to(self.device)
        
        self.adam_alpha = adam_alpha

        self.policy_opt = optim.Adam(
            list(causal_features_encoder.parameters()) + list(policy_network.parameters()), lr=self.adam_alpha
        )

        self.buffer = buffer
        
        self.config = config
        
        
        

    def select_action(self, state, eval_mode=False):
        causal_rep = self.causal_features_encoder(torch.FloatTensor(state).to(self.device))
        action = self.policy_network(causal_rep).argmax()
        action = action.detach().cpu().numpy()

        if eval_mode:
            action = self.policy_network(causal_rep).detach().cpu().numpy()
            num_actions = action.shape[0]
            action = np.argmax(action)
            one_hot_action = np.eye(num_actions)[action]

            action_logits = self.policy_network(causal_rep).detach().cpu().numpy()
            action_prob = softmax(action_logits)
            return one_hot_action, action_prob

        return action

    def train(self, num_updates):
        
        self.curr_update = 1

        for update_index in (range(num_updates)):
            self._update_networks()
            #if update_index % 500 == 0:
            #    print(update_index)
            self.curr_update +=1

        self.env.close()

        self.serialize()

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))

    def _update_networks(self):
        samples = self.buffer.sample()

        loss = self._compute_loss(samples)

        policy_loss = loss 
        
        if self.curr_update % 200 == 0:
            print(self.curr_update, policy_loss)

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()


    def _cross_entropy_loss(self, qvalues, action):
        return nn.CrossEntropyLoss()(qvalues, action)


    def _penalty(self, qvalues, action):
        """
        input: 
            logits: qvalues
            y: actions
        """
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self._cross_entropy_loss(qvalues * scale, action)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)        

        
    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples["state"]).to(self.device)
        action = torch.LongTensor(samples["action"]).to(self.device)
        #next_state = torch.FloatTensor(samples["next_state"]).to(self.device)
        env_ids = torch.LongTensor(samples["env"]).to(self.device)
        

        causal_rep = self.causal_features_encoder(state)  # need this encoder: S -> rep

        # 1. Policy loss
        qvalues = self.policy_network(causal_rep) # need this encoder:  rep -> A
        
        
        ce_loss_env = []
        penalty_env = []
        for idx in range(self.num_training_envs):
            env_idx = torch.where(env_ids==idx)[0]
            env_state = state[env_idx]
            env_action = action[env_idx]
            env_qvalues = qvalues[env_idx]
            
            ce_loss_env.append(self._cross_entropy_loss(env_qvalues, env_action) * len(env_idx))
            penalty_env.append(self._penalty(env_qvalues, env_action) * len(env_idx))
        
        ce_loss = torch.stack(ce_loss_env).sum() / len(env_ids)
        penalty = torch.stack(penalty_env).sum() / len(env_ids)
        
        #ce_loss = self._cross_entropy_loss(qvalues, action)
        #penalty = self._penalty(qvalues, action)
        
        weight_norm = torch.tensor(0.).cuda()
        for w in self.causal_features_encoder.parameters():
            weight_norm += w.norm().pow(2)
        for w in self.policy_network.parameters():
            weight_norm += w.norm().pow(2)
            
            
        loss = ce_loss.clone()
        loss += self.config['l2_regularizer_weight'] * weight_norm
        penalty_weight = (self.config['penalty_weight'] if self.curr_update >= self.config['penalty_anneal_iters'] else 1.0)
        
        """
        if self.curr_update <= 6000:
            penalty_weight = 1
        elif self.curr_update <= 7000:
            penalty_weight = 10
        elif self.curr_update <= 8000:
            penalty_weight = 100
        elif self.curr_update <= 9000:
            penalty_weight = 1000
        else:
            penalty_weight = 10000
        """
        #penalty_weight = 1
        
        #if self.curr_update >= 2001:
        #    penalty_weight = self.curr_update - 2000
        
        loss += penalty_weight * penalty

        if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight


        return loss


