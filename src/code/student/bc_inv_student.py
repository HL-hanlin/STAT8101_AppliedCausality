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
class BCINVStudent(BaseStudent, CUDAAgent):
    def __init__(
        self,
        env,
        trajs_paths,
        model_path,
        num_training_envs,
        teacher,
        causal_features_encoder,
        env_discriminator,
        policy_network,
        buffer,
        adam_alpha,
        config
    ):
        super(BCINVStudent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
            model_path=model_path,
            teacher=teacher,
            buffer=buffer,
        )

        self.num_training_envs = num_training_envs

        self.causal_features_encoder = causal_features_encoder.to(self.device)
        self.env_discriminator = env_discriminator.to(self.device)
        self.policy_network = policy_network.to(self.device)
       

        self.adam_alpha = adam_alpha


        self.rep_optimizer = optim.Adam(
            list(causal_features_encoder.parameters()) + list(policy_network.parameters()), lr=self.adam_alpha,
        )

        self.policy_opt = optim.Adam(
            list(causal_features_encoder.parameters()) + list(policy_network.parameters()), lr=self.adam_alpha
        )

        self.disc_opt = optim.Adam(list(env_discriminator.parameters()), lr=self.adam_alpha)

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

        for update_index in (range(num_updates)):
            ce_loss, disc_entropy, env_discriminator_loss = self._update_networks()
            if update_index % 1000 == 0:
                #print(update_index)
                print('\repoch {}/{}, ce loss {}, entropy {}, env classifier loss {}\t'.format(update_index, num_updates, ce_loss, disc_entropy, env_discriminator_loss ), end="")
        
        print("\n")
        
        self.env.close()

        self.serialize()

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))

    def _update_networks(self):
        samples = self.buffer.sample()

        (ce_loss,
            disc_entropy,
            env_discriminator_loss,
        ) = self._compute_loss(samples)

        rep_loss = disc_entropy 
        policy_loss = ce_loss
        
        #policy_loss = ce_loss + disc_entropy 

        self.rep_optimizer.zero_grad()
        self.policy_opt.zero_grad()
        rep_loss.backward(retain_graph=True)
        policy_loss.backward()
        self.rep_optimizer.step()
        self.policy_opt.step()

        self.disc_opt.zero_grad()
        env_discriminator_loss.backward()
        self.disc_opt.step()
        
        return ce_loss.detach(), disc_entropy.detach(), env_discriminator_loss.detach()



    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples["state"]).to(self.device)
        action = torch.LongTensor(samples["action"]).to(self.device)
        env_ids = torch.LongTensor(samples["env"]).to(self.device)

        causal_rep = self.causal_features_encoder(state)

        # 1. bc loss
        qvalues = self.policy_network(causal_rep)
        ce_loss = nn.CrossEntropyLoss()(qvalues, action)

        # 2. environment classifier entropy loss 
        predicted_env = self.env_discriminator(causal_rep)
        disc_entropy_entropy = torch.mean(F.softmax(predicted_env, dim=1) * F.log_softmax(predicted_env, dim=1))

        # 3. environment classifier
        predicted_env = self.env_discriminator(causal_rep.detach())
        env_discriminator_loss = nn.CrossEntropyLoss()(predicted_env, env_ids)


        return (ce_loss, disc_entropy_entropy, env_discriminator_loss)
