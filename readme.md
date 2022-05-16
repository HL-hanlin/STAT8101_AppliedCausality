This is the repository for the PhD research course STAT 8101 Applied Causality. 

# Abstract of the project

Offline imitation learning aims at training learning agents with collected offline expert demonstrations without further interactions with the environments. Traditional Behavioral Cloning (BC) algorithm in imitation learning uses supervised learning approach to directly map input observations to expert actions, which is prone to causal confusions (spurious correlations) due to the difference among the observed variables from each training environment, and may not be able to generalize well. 

Therefore, it is tempting to ask could we design a method that explicitly learns the underlying causal structure to tackle this issue. However, recent literature shows a sober look that learning disentangled representations is fundamentally impossible without additional information on the models or data. So we need at least some additional information (e.g. multiple environments) to make the causal graph identifiable. 

Greatly inspired by the Nonlinear IRM model proposed by Lu et al. 2019, we consider the setting of learning from expert's demonstrations under multiple environments, with the aim of generalizing well to new unseen environments. We make several adjustments of the original iCaRL three-phase procedure to adapt it to our imitation learning tasks, and proposed our new algorithm, Invariant Behavioral Cloning (IBC). We compare our method against several benchmark methods on three OpenAI Gym control tasks and show its effectiveness in learning imitation policies capable of generalizing to new environments. Finally, to boost our understanding, we also conduct extensive ablation tests over different part of our algorithm, which we believe could inspire future research in the direction of causal imitation learning. 

 
# Records of project process 

Weekly updates of our project is contained in ```journal.md```, which includes related papers, ideas and experiments about this project. Keeping track of these materials is indeed helpful and valuable for reference!


# Code 

Code for our final project is under ```./src/code```. We also includes some additional code we implemented but not included in our final project for VQ-VAEs and VQ-VAE2 on Atari Games under ```./src/unused_code```. 
