import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agent import CUDAAgent

from .base_student import BaseStudent


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def flatten(_list):
    return [item for sublist in _list for item in sublist]


# pylint: disable=arguments-differ
class BCStudent_noconfound(BaseStudent, CUDAAgent):
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
        super(BCStudent_noconfound, self).__init__(
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
        state = state[:-4]
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

        for update_index in (range(num_updates)):
            policy_loss = self._update_networks()
            if update_index % 1000 == 0:
                #print(update_index)
                #print(update_index)
                print('\repoch {}/{}, policy loss {}\t'.format(update_index, num_updates, policy_loss.detach() ), end="")
                
        self.env.close()

        self.serialize()

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))

    def _update_networks(self):
        samples = self.buffer.sample()

        ce_loss = self._compute_loss(samples)

        policy_loss = ce_loss 

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return policy_loss

        
    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples["state"]).to(self.device)
        state = state[:,:-4]
        action = torch.LongTensor(samples["action"]).to(self.device)
        #next_state = torch.FloatTensor(samples["next_state"]).to(self.device)
        #env_ids = torch.LongTensor(samples["env"]).to(self.device)
        
        #print("state", state)
        

        causal_rep = self.causal_features_encoder(state)  # need this encoder: S -> rep

        # 1. Policy loss
        qvalues = self.policy_network(causal_rep) # need this encoder:  rep -> A
        ce_loss = nn.CrossEntropyLoss()(qvalues, action)

        return ce_loss


