# Thinking & Progress

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

* Had a closer look at the paper recent paper [Causal Multi-Agent Reinforcement Learning: Review and Open Problems](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Causal%20MARL%20open%20problems.pdf). Especially the relationship between MDP and MACM(multi-agent causal model, which could be seen as a generalization of SCM), as well as how counterfactuals could be incorporated in decision making of agents.

* For the counterfactual reasoning part, I realized that there had been papers using counterfactual reasoning in MARL starting from 2018 [Counterfactual Multi-Agent Policy Gradients (COMA)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/COMA.pdf). Besides, another paper [Learning to Communicate Using Counterfactual Reasoning](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Vanneste.pdf) further talks about how agents sould communicate informations to improve efficiency. 

* For the MAML side, discovered a good [post](https://yubai.org/blog/marl_theory.html) covers recent progresses in MARL theory, mainly about markov games. Their [new paper](https://arxiv.org/pdf/2110.14555.pdf) which uses adversarial bandits in MARL is also a nice advance as for sample complexity.



<br />
<br />

## Week 4, Feb 21 - Feb 27:

* finished reading the paper [Matrix Completion Methods for Causal Panel Data Models](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/matrix%20completion%20methods.pdf)

* I searched for some more papers this weeks, and found that there are little paper that really uses causal graphs in MARL. The previous papers [COMA](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/COMA.pdf) etc, they defined "counterfactual" without using causal graphs, as the difference in probability distributions. However, we could see a clear connection between structual cansal model(SCM), or [multi-agent causal model (MACM)](https://github.com/HL-hanlin/STAT8100_AppliedCausality/blob/main/etc/reference_papers/Inference%20in%20MACM.pdf) with decentralized MDP. So we should either somehow know the causal graph apriori, or do some structual learning / causal discovery. Therefore, I further read two papers related to causal discovery in RL or MARL problems: [Distributed Learning of Multi-Agent Causal Models](https://ieeexplore.ieee.org/document/1565554), and [Causal Discovery With Reinforcement Learning](https://arxiv.org/pdf/1906.04477.pdf).

* Found another good paper related to counterfactual reasoning on MARL: [Social Influence as Intrinsic Motivation for Multi-Agent Deep RL](https://arxiv.org/pdf/1810.08647.pdf)

* Found a good MARL research team: [Whiteson Research Lab](http://whirl.cs.ox.ac.uk/index.html)


