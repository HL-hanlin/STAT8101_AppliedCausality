This is the repository for the PhD research course STAT 8100 Applied Causality. 

# Abstract of the project

    A key benefit of Multi-Agent Reinforcement Learning (MARL) is the decentralisation of the learning process, which maps to many potential RL applications. While there has been some interest in merging ideas from causality with single-agent systems, no former research explicitly explores the intersection of causality with MARL. 
    
    An interesting real world application is traffic light control. Consider the four interlinked road intersections as shown in the following figure, each traffic light could only observe information about the traffic level on adjacent roads. The goal here is to minimize the total traffic of the roads (joint reward). If we further assume history is shared among traffic lights, then a way to model this is through decentralized Partially Observed Markov Decision Process (Dec-POMDP). 
    
    Since each traffic light could receive information both from observations $\mathcal{L}_1$ and interventions $\mathcal{L}_2$, it is interesting to think if it is possible to incorporate ideas from causality into the modeling of this problem. I'm interested in studying how knowledge could be shared across these traffic lights, since communicating causal relationships could probably boost learning efficiency in each traffic light (or in each decentralised tasks to be more general and not example-specific). 
    
    Another realistic and interesting setting I could think of is to assume one (or more) traffic light has better sensor that could gather information across several blocks, while other traffic lights could only sense locally. In such setting, we could probably borrow ideas from causal transfer learning and causal imitation learning.

