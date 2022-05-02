This is the repository for the PhD research course STAT 8100 Applied Causality. 

# Abstract of the project

Offline imitation learning aims at using collected expert demonstrations to train learning agents. Traditional Behavioral Cloning (BC) algorithm in imitation learning directly maps input observations to expert actions, which is prone to spurious correlations (sometimes also called as causal confusions) due to the difference of observed variables from each training environment, and may not be able to generalize well. Therefore, it is tempting to ask could we design a method that explicitly learns the underlying causal structure to tackle this issue. However, recent literature shows a sober look that unsupervised learning disentangled representations is fundamentally impossible without additional information on the models or data. So we need at least some additional information (e.g. multiple environments) to make the causal graph identifiable. 

Greatly inspired by the Nonlinear IRM model proposed by Lu et al., we consider the setting of learning from expert’s demonstrations from multiple environments, with the aim of generalizing well to new unseen environments. We make several adjustments of the original iCaRL three-phase procedure to adapt it to our imitation learning tasks, and proposed our new algorithm, Invariant Behavioral Cloning (IBC). We compare our method against several benchmarks on three OpenAI Gym control tasks and show its effectiveness in learning imitation policies capable of generalizing to new environments. Finally, to boost our understanding, we also conduct extensive ablation tests over different part of our algorithm, which we believe could inspire the direction of future research in causal imitation learning.

 
