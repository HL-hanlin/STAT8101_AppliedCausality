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
class CCILStudent(BaseStudent, CUDAAgent):
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
        super(CCILStudent, self).__init__(
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mask_prob = config['MASK_PROB']
        
        self.best_mask = None
        self.best_loss = np.inf
        

    def select_action(self, state, eval_mode=False):
        
        state = torch.FloatTensor(state).to(self.device)
        prob = torch.ones(state.size()) 
        mask = torch.bernoulli(prob).to(self.device)
        #mask = self.best_mask[0,:]
        #state_concat = torch.cat([state * mask, mask], dim=0)[None,:]
        state_concat = (state * mask)[None,:]
        
        causal_rep = self.causal_features_encoder(state_concat)

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
        loss_list = []
        for update_index in (range(num_updates)):
            policy_loss = self._update_networks()
            if update_index % 1000 == 0:
                #print(update_index)
                print('\repoch {}/{}, policy loss {} \t'.format(update_index, num_updates, policy_loss.detach() ), end="")
            loss_list.append(policy_loss.detach())
        

        self.env.close()

        self.serialize()
        
        return loss_list

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))

    def _update_networks(self):
        samples = self.buffer.sample()

        ce_loss, mask = self._compute_loss(samples)
        
        #if ce_loss.detach() < self.best_loss:
        #    self.best_mask = mask
        #    self.best_loss = ce_loss.detach()
        #    print("best mask {}, best loss {}".format(mask[0,:], self.best_loss))

        policy_loss = ce_loss 

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return policy_loss

        
    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples["state"]).to(self.device)
        action = torch.LongTensor(samples["action"]).to(self.device)
        #next_state = torch.FloatTensor(samples["next_state"]).to(self.device)
        #env_ids = torch.LongTensor(samples["env"]).to(self.device)
        
                
        prob = torch.ones(state.size()) * (1 - self.mask_prob)
        mask = torch.bernoulli(prob).to(self.device)
        
        #prob = torch.ones(state.size()) * (1 - self.mask_prob)
        #mask = torch.bernoulli(prob).to(self.device)
        #mask = torch.tile(mask, (state.shape[0],1))
        

        #state_concat = torch.cat([state * mask, mask], dim=1)
        state_concat = state * mask

        causal_rep = self.causal_features_encoder(state_concat)  # need this encoder: S -> rep

        # 1. Policy loss
        qvalues = self.policy_network(causal_rep) # need this encoder:  rep -> A
        ce_loss = nn.CrossEntropyLoss()(qvalues, action)

        return ce_loss, mask


