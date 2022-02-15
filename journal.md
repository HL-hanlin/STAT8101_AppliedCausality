# Thinking & Progress

## Week 1, Jan 23 - Jan 30: 

Created this repo!

Finished the assigned reading: [Causal Inference In Statistics: A Primer](https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf). 

Detailed [reading notes](https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf) is under /doc folder of this repo.


## Week 2, Jan 31 - Feb 6:

Start thinking about project topics. 

I took IEOR 4575 Reinforcement Learning from Prof. Shipra, as well as COMS 6998 Bandits & Reinforcement Learning from [Prof. Krishnamurthy](https://people.cs.umass.edu/~akshay/) this semester, so I'm very willing to do some projects at the intersection of causal inference and reinforcement learning!

My strategy this week is try to read broadly to find a potential good topic for research! 

This [Causal Reinforcement Learning (CRL)](https://crl.causalai.net/) website is quite useful, since it summarizes recent advances of CRL according to several tasks pretty systematically. I started my reading from several papers that utilize causal inference on [Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit), which is an important RL model.

* I first read the paper [Bandits with Unobsergved Confounders: A Causal Approach (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/Bandits%20with%20Unobserved%20Confounders.pdf), which is probably the paper that intersects causal inference with RL models with unobserved confounders (UC). It showed that when UC exists, current bandit algorithms which try to maximize rewards based on estimation of the experimental distribution, are not always the best to pursue. The greedy casino example in this paper is quite useful for illustration.

* To follow up, I read the other paper [Counterfactual Data-Fusion for Online Reinforcement Learning (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/Counterfactual%20Data-Fusion%20for%20Online%20Reinforcement%20Learners.pdf) that could be seen as a generalization of the greedy casino example in the last paper by using counterfactual-based decision making. 

* The above two papers mainly deal with unobserved confounders, while the following paper [Transfer Learning in Multi-Armed Bandits: A Causal Approach (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/Transfer%20Learning%20in%20Multi-Armed%20Bandits.pdf) talks about how to transfer knowledge across bandit agents where causal effects cannot be identified by do-calculous.


Next, I also find the research at the intersection of imitation learning (reverse reinforcement learning) and causal inference quite interesting. Recent Papers






## Week 3, Feb 7 - Feb 13:

