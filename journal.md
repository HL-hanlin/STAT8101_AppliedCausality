# Thinking & Progress

<br />

This markdown file keeps track of our project prograss. At the first several weeks, I mainly focus on the problem of Multi-agent Reinforcement Learning (MARL), where counterfactual thinking could be an intersting problem. Then I gradually switch to the topic of causal confusion in imitation learning, which is the backup project plan I had besides MARL. Greatly inspired by the non-linear IRM paper from Lu et al. we read in class, I finally decide to focus specifically on causal confusion with multiple environments. 

Therefore, I mainly focus on reading related papers in the first several weeks, and then gradually conduct more experiments.


<br />
<br />


## Week 1, Jan 23 - Jan 30: 

* Created this repo!

* Finished the assigned reading: [Causal Inference In Statistics: A Primer](https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf). Detailed [reading notes](https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf) is under /doc folder of this repo.

I first tried to read this book without taking any notes. But after going through it once, I found that I did not learn the concepts & materials very well, and realized that it might be better for me to make some notes by summarizing the key points in this book to help my understanding. 

Therefore, I nearly spent the whole week making the very long reading notes as above (which is obvious not readable for other students, so I will not do this next time :) 

But on the other hand, such notes is a good reference. And I indeed looked it back several times to review some definitions & properties during this whole semester. 



<br />
<br />

## Week 2, Jan 31 - Feb 6:

Start thinking about project topics. 

I took IEOR 4575 Reinforcement Learning from Prof. Shipra, as well as COMS 6998 Bandits & Reinforcement Learning from [Prof. Krishnamurthy](https://people.cs.umass.edu/~akshay/) this semester, so I'm very willing to do some projects at the intersection of causal inference and reinforcement learning!

<br />

My strategy this week is try to read broadly to find a potential good topic for research! 

<br />

This [Causal Reinforcement Learning (CRL)](https://crl.causalai.net/) website is quite useful, since it summarizes recent advances of CRL according to several tasks pretty systematically. I started my reading from several papers that utilize causal inference on [Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit), which is an important RL model.

* I first read the paper [Bandits with Unobsergved Confounders: A Causal Approach (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Bandits%20with%20Unobserved%20Confounders.pdf), which is probably the paper that intersects causal inference with RL models with unobserved confounders (UC). It showed that when UC exists, current bandit algorithms which try to maximize rewards based on estimation of the experimental distribution, are not always the best to pursue. The greedy casino example in this paper is quite useful for illustration.

* To follow up, I read the other paper [Counterfactual Data-Fusion for Online Reinforcement Learning (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Counterfactual%20Data-Fusion%20for%20Online%20Reinforcement%20Learners.pdf) that could be seen as a generalization of the greedy casino example in the last paper by using counterfactual-based decision making. 

* The above two papers mainly deal with unobserved confounders, while the following paper [Transfer Learning in Multi-Armed Bandits: A Causal Approach (annotated)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Transfer%20Learning%20in%20Multi-Armed%20Bandits.pdf) talks about how to transfer knowledge across bandit agents where causal effects cannot be identified by do-calculous.


<br />
<br />


## Week 3, Feb 7 - Feb 13:


Next, I also find the research at the intersection of imitation learning (reverse reinforcement learning) and causal inference quite interesting. The RL course I took this semester also covers Imitation Learning [(this is the syllabus)](https://people.cs.umass.edu/~akshay/courses/coms6998-11/index.html) as a topic (but will only touch it nearly at the end of the semester). To confess, I did not have enough time to read the two papers I listed below carefully. But I will definitely go back to this topic later in this semester!

* Recent Papers like [Causal Imitation Learning with Unobserved Confounders](https://causalai.net/r66.pdf) and [Sequential Causal Imitation Learning with Unobserved Confounders](https://causalai.net/r76.pdf) nicely studies UC problems in imitation learning. 

* Prof. [Ermon](https://cs.stanford.edu/~ermon/) has also done several research related to imitation learning (but only from the RL side). Maybe I could also read some of his paper to see if I could find some good idea.

<br />

Then I also searched for topics related to Multi-Agent Reinforcement Learning, and actually find a quite good summary!

* This very recent paper [Causal Multi-Agent Reinforcement Learning: Review and Open Problems](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Causal%20MARL%20open%20problems.pdf) summarizes several open problems in causal MARL, and it mentioned that researchers before did not make enough connections between the CI side and the MARL side. A good synthetic example it listed there is the traffic light control problem, which is a good example to illustrate the usefullness of causal inference on MARL problems. And I think this might be a good way to delve into in the following weeks, since it is an active and valuable area of research! :)


<br />
<br />


## Week 4, Feb 14 - Feb 20:

* finished reading the paper [Using Synthetic Controls](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Using%20Synthetic%20Controls.pdf)

* Had a closer look at the paper recent paper [Causal Multi-Agent Reinforcement Learning: Review and Open Problems](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Causal%20MARL%20open%20problems.pdf). Especially the relationship between MDP and MACM(multi-agent causal model, which could be seen as a generalization of SCM), as well as how counterfactuals could be incorporated in decision making of agents.

* For the counterfactual reasoning part, I realized that there had been papers using counterfactual reasoning in MARL starting from 2018 [Counterfactual Multi-Agent Policy Gradients (COMA)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/COMA.pdf). Besides, another paper [Learning to Communicate Using Counterfactual Reasoning](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Vanneste.pdf) further talks about how agents sould communicate informations to improve efficiency. 

* For the MAML side, discovered a good [post](https://yubai.org/blog/marl_theory.html) covers recent progresses in MARL theory, mainly about markov games. Their [new paper](https://arxiv.org/pdf/2110.14555.pdf) which uses adversarial bandits in MARL is also a nice advance as for sample complexity.



<br />
<br />

## Week 5, Feb 21 - Feb 27:

* finished reading the paper [Matrix Completion Methods for Causal Panel Data Models](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/matrix%20completion%20methods.pdf)

* I searched for some more papers this weeks, and found that there are little paper that really uses causal graphs in MARL. The previous papers [COMA](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/COMA.pdf) etc, they defined "counterfactual" without using causal graphs, as the difference in probability distributions. However, we could see a clear connection between structual cansal model(SCM), or [multi-agent causal model (MACM)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Inference%20in%20MACM.pdf) with decentralized MDP. So we should either somehow know the causal graph apriori, or do some structual learning / causal discovery. Therefore, I further read two papers related to causal discovery in RL or MARL problems: [Distributed Learning of Multi-Agent Causal Models](https://ieeexplore.ieee.org/document/1565554), and [Causal Discovery With Reinforcement Learning](https://arxiv.org/pdf/1906.04477.pdf).





<br />
<br />

## Week 6, Feb 28 - Mar 6:

* Found another good paper related to counterfactual reasoning on MARL: [Social Influence as Intrinsic Motivation for Multi-Agent Deep RL](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Social%20Influence.pdf). This paper introduces a games where multiple agent will only maximize their total reward if they choose to collaborate. To achieve collaboration, the paper uses counterfactual reasoning to calculate intrinsic motivation for each agent, and includes it as part of the reward for each agent. The paper uses KL divergence between the marginal distribution and conditional distribution to measure the counterfactuals. This idea seems to be quite general, and it's also quite nice that we could formulate the causal graph of this problem (even in different ways). Here are some extentions I could think of based on this paper:
  * From the simulation video for cleanup task, it seems that if the green apple appears on the rightmost edge and the purple agent moves there to collect the apple, then this agent will just stop there without moving back (this is based on the assumption that the purple agent is lazy to move). However, this will make this purple agent unobservable from the pink agent, which will lead to pink agent also comming to collect apples rather than mining for apples. I think changing the coefficients $\alpha$ and $\beta$ before the external and intrinsic rewards might be able to solve this issue. Or there might be a way to make these two parameters dynamic. 
  * Can we add unobserved confounders into the causal diagram? (like the paper MDP with unobserved confounders: a causal approach)
    * Update: It seems that POMDP and MDPUC are "orthogonal" to each other.
  * It seems that such formulation might also work for other experiment set ups (for example, the traffic lights coorporation example, since each traffic light could also only observe part of the states). 
  * Some other related papers from the same group: [Intrinsic Social Motivation via Causal Inference on MARL](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/intrinsic%20social%20motivation.pdf), and [Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Inequity%20aversion%20improves%20cooperation%20in%20intertemporal%20social%20dilemmas.pdf)
  * Two main authors: [eugenevinitsky](https://eugenevinitsky.github.io/), [natashajaques](https://natashajaques.ai/), and code [here](https://github.com/eugenevinitsky/sequential_social_dilemma_games)

* Found some good MARL research teams: [Whiteson Research Lab](http://whirl.cs.ox.ac.uk/index.html), [Wen Sun](https://wensun.github.io/), [Berkeley MARL Seminar](https://sites.google.com/view/berkeleymarl/home)







<br />
<br />

## Week 7, Mar 7 - Mar 13:

* Mainly Preparing for midterms this week. 

* Tried to implement the code from [Social Influence as Intrinsic Motivation for Multi-Agent Deep RL](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Social%20Influence.pdf). However, I found that their code need pretty large computational resource, which is not so practical to run on my PC. Then I looked into several other MARL papers, and realized that papers in this field are usually all very computational intensive, which makes me hard to proceed.







<br />
<br />

## Week 8, Mar 14 - Mar 20:

### Causal Confusion in Imitation Learning


In this week, I decided to switch to the backup project of causal confusion in imitation learning. The main reason for this is that I found the paper [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf) really interesting! 

* Read the paper [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf).

* Implemented the code for [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf).


In this paper, they propose a non-causal approach to solve the causal confusion problem. Therefore, a natural question is: can we design some method that could solve the causal confusion problem by learning the underlying causal graph explicitly?

With such question in mind, I also searched and found the following two papers might be relevant.

* Read the paper [CausalVAE: Disentangled Representation Learning
via Neural Structural Causal Models](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CausalVAE_Disentangled_Representation_Learning_via_Neural_Structural_Causal_Models_CVPR_2021_paper.pdf). It assumes that we already know n concepts in the image data (e.g. 4 concepts: smile, age, gender, haircolor). However, we do not have these concepts in our setting. Could we regard these n concepts as the K different categories in the codebook on our VQ-VAE model?

* Read the paper [Discovering Causal Signals in Images](https://arxiv.org/pdf/1605.08179.pdf). It provides a procedure to detect objects in an image scene (e.g. detact wheels from cars). And it only works for such object-scene pairs. This is a little bit different from our setting: we want to know the causal relationship among K different categories in our codebook of VQ-VAE. Our setting is more complicated: there might be possible that more than one categories together will cause another one or more category/categories. So we could not use their methods directly. This strengthened our question: are there better causal discovery methods better than methods based on random dropout?






<br />
<br />

## Week 9, Mar 21 - Mar 27:

* In this week, I mainly focus on implementing different VAEs. I mainly follow the paper [Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning](https://arxiv.org/pdf/2110.14118.pdf) and tried their VQ-VAEs on Pong Atari Games, which could achieve much better scores than beta-VAEs as in paper [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf). 

* I found that besides VQ-VAEs, people also propose more accurate models including [VQ-VAE2](https://arxiv.org/abs/1906.00446) as well as even more fancy model that includes attentions to VAEs. I coded VQ-VAE2 and tested it on the Pong game, but the performance is not significantly different from VQ-VAE. So I did not continue in this direction. (I guess VAEs should also not be a focus in the project, and we should concentrate more on causal inference itself).







<br />
<br />

## Week 10, Mar 28 - Apr 3:

* Briefly discussed my project idea with Prof. Blei. I got really stucked at this point since it seems that the yellow indicator light in [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf) is perfectly correlated with brakes, so there's no way to know indicator light is the effect rather than the cause of brake. As suggested by Prof., multiple environments might be a way to direction to go, since unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both the models and the data as proved by Locatello in the paper [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359). 

* Therefore, I focus on the Atari Games dataset (especially Pong), and tried to create images from multiple environments. My procedure for creating 2 different environments are as follows: in the first environment, I just use the original image from games screenshot. In the second environment, I masked out the scores at the top of each image, and added a number representing precious actions at the left bottom corner of the image. In this way, the first image could represent an environment with scores (which is a effect of action), and the second image could represent an environment with previous actions (which is cause of action). We use these two variants as our training environments, and we define our testing environment as the image with neither scores nor previous actions (so it is not contaminated by spurious informations). The illustrative image is [here!](https://github.com/HL-hanlin/STAT8101_AppliedCausality/blob/main/etc/Pong_traineval2.png)

* However, I'm not quite sure how to create multiple environments for other Atari Games, since the scores at the top of the image is unique to this Pong dataset. 









<br />
<br />

## Week 11, Apr 4 - Apr 10:

* This week I deviates from Atari Games and tried to search for other datasets. I found the paper [Causal Imitative Model for Autonomous Driving](https://github.com/vita-epfl/CIM) pretty interesting. It addresses inertia and collision problems in self-driving by discovering the causal graphs and utilizes it to train the policy. 

* Then I started to reproduce their results. They did not provide the dataset, but only with the code to generate these datasets. However, I realized that even these code are also flawed since some important hyperparameters are not disclosed. I tried to guess these hyperparameters and generated some simulated datasets, but non of my trials could achieve the same results as provided in their paper. Pretty sad this week :(





<br />
<br />

## Week 12, Apr 11 - Apr 17:

* Prof. Blei talks about the paper [Nonlinear Invariant Risk Minimization: A Causal Approach](https://arxiv.org/abs/2102.12353) this week. This paper is a turning point of my project progress! 

* I did something crazy to look into each paper that cites [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf) (around 130 papers in total!) in order to find some new idea about how to construct multiple environments. Luckily, I found the paper [Invariant Causal Imitation Learning for Generalizable Policies](https://openreview.net/forum?id=715E7e6j4gU) which contains a nice way to create spurious correlates into the OpenAI Gym Tasks. 

* Then I started the process of implementing the iCaRL algorithm in [Lu et al.](https://arxiv.org/abs/2102.12353) with the multiple environments settings in this paper. 





<br />
<br />

## Week 13, Apr 18 - Apr 24:

* I got stucked in phase 1 for a long time, especially how to implement the non-factorized prior. I initially thought that the latent states X is unobservable, so it is wierd to use it as input to calculate the non-factorized prior as shown in appendix I.2 of Lu et al. Finally, I realized that X is from the encoder which is predicted from the observations, so this is not an issue.

* I found some open source implementation of HSIC and KCIT tests in phase 2. However, these methods is pretty slow and memory consuming. It works for sample size of 5000, but not for 50000 (which is the size I need in my experiment). Therefore, I start to search for other methods, including [Causal Inference on Discrete Data using Additive Noise Models](https://arxiv.org/abs/0911.0280#:~:text=Inferring%20the%20causal%20structure%20of,the%20case%20of%20continuous%20variables.) and [Distribution-Free Learning of Bayesian Network Structure in Continuous Domains](https://www.aaai.org/Papers/AAAI/2005/AAAI05-130.pdf). 

* The method in [Distribution-Free Learning of Bayesian Network Structure in Continuous Domains](https://www.aaai.org/Papers/AAAI/2005/AAAI05-130.pdf) seems reasonable to me, so I started to implement their algorithm on my own (because I didn't find open source implementation of their algorithm). 
   * Sadly, I think the sample size I need is still too large for this testing method, since it needs to calculate \Gamma(n), and n here could be as large as 50000. I tried several trick in my code to avoid computation of such large number, but it still couldn't work. 





<br />
<br />

## Week 19, Apr 25 - May 1:


* Realizing score matching could lead to great variance, I searched for a while and found several methods are designed to tackle this problem, including DCM and SSM. From the paper of [Sliced Score Matching](https://arxiv.org/pdf/1905.07088.pdf), which is published in year 2019, it achieves the best performance in comparison with other methods. Therefore, we adopt their method in replace of the original score matching in phase 1 from Lu et al 2019. From the training loss curve, we indeed see a great benefit from using SSM! The loss curve stabilizes after around 500 iterations. In comparison, we need to choose the trained model which achieves the lowest training loss in our original score matching implementation, which uas a very large variance. I'm pretty happy with this SSM method which indeed helps a lot!

* Thanks David for his suggestions in Slack, I tried the [FCIT](https://github.com/kjchalup/fcit) algorithm in phase 2, which works well! 

* After finishing implementation of phase 2, I proceed to phase 3 this week. However, I found the optimization procedure (Equation 12 in Lu et al.) is too time consuming and not so practical. It takes 100X more time than training a new mapping (neural networks) from observations to latent states. 

* To avoid optimizing equation 12, I tried two approaches: 
   * (1) for a new observational data point, we calculates its L2 distance with the observation data points in our training set. We could use the latent states from the observational data point with smallest L2 distance as a proxy for the latent states of this new data point. 
   * (2) learn another mapping from observations to latent states. 

I found that method (2) works better in practice because we could slightly perturb training data points and general more fake data used for training, which could make the our neural network more robust in prediction corresponding latent states. 






<br />
<br />

## Week 15, May 2 - May 8:

* As suggested by Prof. Blei during the office hour, we need to correct for multiplicity when using multiple independent FCIT tests in phase 2 of our IBC algorithm. The paper [The multiple problems of multiplicity - whether and how to correct for many statistical tests](https://pubmed.ncbi.nlm.nih.gov/26245806/) is a pretty good summary that contains several methods for doing multiplicity correction. I followed the method Sidak-Bonferroni to adjust the confidence level \alpha to correct for multiplicity. 

* However, I gradually realized that such confidence level \alpha is also a hyper-parameter that needs to be carefully tuned. In our original method, we define a threshold T. If a latent variable $Z_i$ is tests to be independent with k other latent variables, and the value for k is >= T, then we regard it as one of the direct parents of actions A. So the threshold T is a hyper-parameter that needs to be tuned in our original method. Comparing these old and new approaches, since we always need to tune some hyper-parameter, I guess there is no significant benefit from correction of multiplicity. 






<br />
<br />

## Week 16, May 9 - May 15:

This week, I mainly focus on writing the project report (as well as preparing for final exams from our courses). 







