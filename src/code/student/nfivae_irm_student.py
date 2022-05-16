import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch import autograd
import copy
import random

from torch.utils.data import DataLoader

from agent import CUDAAgent
from vae.ivae import iVAE
from vae.nfivae import NFiVAE

from network import StudentNetwork

from .base_student import BaseStudent

try:
    from testing.paths import get_trajs_path  # noqa
except (ModuleNotFoundError, ImportError):
    from .testing.paths import get_trajs_path  # pylint: disable=reimported



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def flatten(_list):
    return [item for sublist in _list for item in sublist]



class Phase3ObstoLatentEncoder(nn.Module):
    def __init__(self, obs_size, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64,  32),
            nn.ELU(),          
            nn.Linear(32, latent_size))

    def forward(self, x):
        return self.layers(x)






# pylint: disable=arguments-differ
class NFiVAE_IRMStudent(BaseStudent, CUDAAgent):
    def __init__(
        self,
        env,
        vae_wrapper,
        trajs_paths,
        model_path,
        num_training_envs,
        teacher,
        buffer,
        adam_alpha,
        config,
        #phase3_obs_to_latent_encoder = None,
        causal_features_encoder = None, 
        policy_network = None
    ):
        super(NFiVAE_IRMStudent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
            model_path=model_path,
            teacher=teacher,
            buffer=buffer,
        )

        self.num_training_envs = num_training_envs

        self.adam_alpha = adam_alpha
        self.causal_features_encoder = causal_features_encoder.to(self.device)
        self.policy_network = policy_network.to(self.device)
        
        #self.phase3_obs_to_latent_encoder = phase3_obs_to_latent_encoder.to(self.device) # only for phase 3!!!
        #self.phase3_obs_to_latent_encoder = None
        
        self.policy_opt = optim.Adam(  list(policy_network.parameters()), lr=self.adam_alpha)
        

        self.buffer = buffer
        
        self.config = config
        
        self.vae_wrapper = vae_wrapper
        
        self.selected_states = self.vae_wrapper.pa_list
        self.masked_states = list(set(np.arange(self.vae_wrapper.latent_dim)).difference(self.vae_wrapper.pa_list))
        
        self.latent_norm = None

        print("policy network", self.policy_network)

    


    
    ############################        phase 3.       #########################
    ############################        phase 3.       #########################
    ############################        phase 3.       #########################


    def train(self, num_updates):
        print("train in NFiVAE")
        policy_loss_list = []
        for update_index in (range(num_updates)):
            policy_loss, obs_to_latent_loss = self._update_networks()
            policy_loss_list.append( policy_loss.detach() )
            if update_index % 1000 == 0:
                print('\repoch {}/{}, policy loss {}\t'.format(update_index, num_updates, policy_loss.detach() ), end="")
                
        return policy_loss_list
               
        self.env.close()

        self.serialize()

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))



    def _update_networks(self):
        samples = self.buffer.sample()

        ce_loss, mse_loss = self._compute_loss(samples)

        policy_loss = ce_loss 
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        
        obs_to_latent_loss = mse_loss
        """
        self.obs_to_latent_opt.zero_grad()
        obs_to_latent_loss.backward()
        self.obs_to_latent_opt.step()
        """
        
        return policy_loss, obs_to_latent_loss
        




  
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
        env_ids = torch.LongTensor(samples["env"]).to(self.device)

        latents = self.vae_wrapper.predict_latent(samples).detach()
        
        
        """
        ###################################################################
        # phase 3. obs_to_latent  MSE loss
        obs_to_latent_rep = self.phase3_obs_to_latent_encoder(state)
        mse_loss = nn.MSELoss()(obs_to_latent_rep, latents)
        ###################################################################
        """
        mse_loss = None
        
        ###################################################################
        # phase 1. policy loss
        masked_latents = copy.deepcopy(latents)
        #if len(self.masked_states)>0:
        #    masked_latents[:, self.masked_states] = 0
        
        masked_latents = latents[:,self.vae_wrapper.pa_list]
        
        #causal_rep = self.causal_features_encoder(masked_latents)  # need this encoder: S -> rep
        #qvalues = self.policy_network(causal_rep) # need this encoder:  rep -> A
        qvalues = self.policy_network(masked_latents) # need this encoder:  rep -> A
        
        
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
        
        
        weight_norm = torch.tensor(0.).cuda()
        for w in self.policy_network.parameters():
            weight_norm += w.norm().pow(2)        
        
        loss = ce_loss.clone()
        loss += self.config['l2_regularizer_weight'] * weight_norm
        
        penalty_weight =  self.config['penalty_weight']
        loss += penalty_weight * penalty
    
        if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        #ce_loss = nn.CrossEntropyLoss()(qvalues, action)

        return loss , mse_loss













    
    ############################        phase 3.       #########################
    ############################        phase 3.       #########################
    ############################        phase 3.       #########################

        
    def generating_stored_data(self):
        
        
        all_states = []
        all_actions = []
        all_envs = []
    
        trajs_paths = []
        for env_id in range(self.config["NUM_TRAINING_ENVS"]):
            path = get_trajs_path(self.config["ENV"], "expert", env_id)
            trajs_paths.append(path)
    
        for path in trajs_paths:
            trajs = np.load(path, allow_pickle=True)[()]["trajs"]
    
            for traj in trajs:
                for i in range(len(traj)-1):
                    all_states.append(traj[i][0])
                    all_actions.append(traj[i][1])
                    all_envs.append(traj[i][2])
    
        x = torch.FloatTensor(all_states).to(self.device)
    
        a = F.one_hot(torch.LongTensor(all_actions), num_classes = self.vae_wrapper.action_dim)#.to(self.device)
              
    
        if self.vae_wrapper.use_a and self.vae_wrapper.use_e:
            print("generating stored samples using both a and e")
            e = F.one_hot(torch.LongTensor(all_envs), num_classes = self.vae_wrapper.env_dim)#.to(self.device)
            ae = torch.cat((a, e), 1).to(torch.float32).to(self.device)
        elif self.vae_wrapper.use_e == False:
            print("generating stored samples using only a")
            ae = a.to(torch.float32).to(self.device)
    
        _, _, z, _ = self.vae_wrapper.model(x, ae)
    
        self.stored_x = x.detach().cpu().numpy()
        self.stored_z = z.detach().cpu().numpy()
        self.stored_a = a.detach().cpu().numpy()
        
        print("using all data to generate stored data", self.stored_x.shape)
        
        """
        
        print("using sampled data to generate stored data")
        
        self.stored_x, self.stored_z, _ = self.vae_wrapper.phase2_sampled_data(n_samples = 1000000)
        
        print("generated stored_x, stored_z with shape = ", self.stored_x.shape)
        
        #return x, z
        """


    """
    def infer_latent_from_observation(self, state_env, phase3_latent_samples = 100000, lambda_1 = 10, lambda_2 = 0.01, lr = 0.005, num_iters = 750 ):
        #lambda_1 = 10
        x_single = torch.FloatTensor(state_env)[None,:].to(self.device)
        Z_opt = torch.zeros(self.vae_wrapper.latent_dim, requires_grad=True)
        
        opti = optim.Adam([Z_opt], lr=lr)
        
        for i in range(num_iters):
            decoder_params = self.vae_wrapper.model.decoder_params(Z_opt.to(self.device))
            log_px_z = self.vae_wrapper.model.decoder_dist.log_pdf(x_single, *decoder_params)
        
            loss =  -( log_px_z)  + lambda_1 * torch.norm(Z_opt)**2 #+ lambda_2 * torch.norm(Z_opt[mask_list])**2  #)
            opti.zero_grad()
            loss.backward(retain_graph=True)
            opti.step()
            
            #if i % 50 ==0:
            #    print(i, loss, torch.norm(Z_opt), Z_opt)
        
        Z_opt = Z_opt / torch.norm(Z_opt) * self.latent_norm
        
        return Z_opt.detach().to(self.device)
    """
                

    
    """
    def find_matched_z(self, x, dim_list = None):
        if dim_list == None:
            dim_list = np.arange(self.vae_wrapper.state_dim).tolist()
        return self.stored_z[np.argmin(np.linalg.norm(x[dim_list] - self.stored_x[:, dim_list], axis = 1))]
    """
        
        
        

    def gen_phase3_obs_to_latent_encoder(self, phase3_encoder_lr = 0.01, phase3_bs = 1024, phase3_epochs = 50):
        
        print(" *** start training phase3 obs->latent encoder *** ")
        
        #phase3_encoder = phase3_obs_to_latent_encoder(self.vae_wrapper.state_dim, self.vae_wrapper.latent_dim).to(self.device)
        phase3_encoder = Phase3ObstoLatentEncoder(self.vae_wrapper.state_dim, len(self.vae_wrapper.pa_list)).to(self.device)
        
        phase3_optimizer = torch.optim.Adam(phase3_encoder.parameters(), lr=phase3_encoder_lr)

        total_idxs = list(range(len(self.stored_z)))
        n_batch = len(self.stored_z) // phase3_bs + 1
        phase3_loss_list = []

        for epoch in tqdm(range(phase3_epochs)):
            random.shuffle(total_idxs)
            phase3_epoch_loss = 0
            for j in range(n_batch):
                batch_idxs = total_idxs[j * phase3_bs : (j + 1) * phase3_bs]
                batch_x = torch.as_tensor(self.stored_x[batch_idxs], device=self.device)#.long()
                batch_z = torch.as_tensor(self.stored_z[batch_idxs], device=self.device)#.long()
                masked_batch_z = batch_z[:,self.vae_wrapper.pa_list]
                
                phase3_optimizer.zero_grad() 
                z_pred = phase3_encoder(batch_x)
                #phase3_loss =  nn.L1Loss()(z_pred, batch_z)
                phase3_loss =  nn.L1Loss()(z_pred, masked_batch_z)
                phase3_loss.backward() 
                phase3_optimizer.step() 
                phase3_epoch_loss += phase3_loss.detach().cpu().item()

            print(epoch, phase3_epoch_loss)

        self.phase3_obs_to_latent_encoder = phase3_encoder




    
    ############################        evaluation.       #########################
    ############################        evaluation.       #########################
    ############################        evaluation.       #########################


    def select_action(self, state, eval_mode=False):
        
        #inferred_latent = torch.FloatTensor(self.find_matched_z(state, self.config['dim_list'])).to(self.device)
        inferred_latent = self.phase3_obs_to_latent_encoder.forward(torch.FloatTensor(state).to(self.device))

        #print("inferred_latent", inferred_latent)


        #if len(self.masked_states)>0:
        #    inferred_latent[self.masked_states] = 0 # added this line!
        
        #inferred_latent_detached = inferred_latent.detach().cpu()
        #inferred_latent_detached = inferred_latent_detached[self.vae_wrapper.pa_list]
        #causal_rep = self.causal_features_encoder(inferred_latent)  # need this encoder: S -> rep
        #action = self.policy_network(causal_rep).argmax()
        
        #print("inferred_latent_detached", inferred_latent_detached)
        
        #action = self.policy_network(inferred_latent_detached.to(self.device)).argmax()
        action = self.policy_network(inferred_latent).argmax()
        
        
        #print("action", action)
        
        action = action.detach().cpu().numpy()

        if eval_mode:
            action = self.policy_network(inferred_latent).detach().cpu().numpy()
            num_actions = action.shape[0]
            action = np.argmax(action)
            one_hot_action = np.eye(num_actions)[action]

            action_logits = self.policy_network(inferred_latent).detach().cpu().numpy()
            action_prob = softmax(action_logits)
            return one_hot_action, action_prob

        return action
        
        