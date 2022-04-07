# Report for Assignment 3

### 519021910913 黄喆敏

### Directory Structure

````
lab3
├── code
│   ├── GridWorld.py	
│   ├── MC.py											# Monte-Carlo Methods
│   └── TD.py											# TD(0) Method
└── docs
    ├── assets										# images for report
    │   ├── TD_policy.png
    │   ├── TD_value.png
    │   ├── every_visit_policy.png
    │   ├── every_visit_value.png
    │   ├── first_visit.jpg
    │   ├── first_visit_policy.png
    │   ├── first_visit_value.png
    │   └── td0.jpg
    ├── report.md
    └── report.pdf
````

All codes are placed in `./code` directory. To run the code, you can either run `MC.py`, or `TD.py`, in which I have implemented two methods separately.



### Cliff Walking

 $\epsilon-greedy$ policy is a simple way to balance exploration and exploitation by choosing exploration and exploitation randomly.  For $\epsilon-greedy$ policy,
$$
\pi(a|s)=\begin{cases} \frac{\epsilon}{m}+1-\epsilon，if\ a^*={argmax}_{a\in A}  \\ \frac{\epsilon}{m}，\ \ \ \ \ \ \ \ \ \ \ \ \   otherwise\end{cases}
$$






We use $\epsilon-greedy$ policy to choose action in both SARSA and Q-Learning method, and the implementation is mentioned below.





### SARSA Method

For SARSA method, the algorithm can be given as follows. It is important that while updating the policy, the same policy is used.

![sarsa](./assets/sarsa.png)



### Q-Learning Method

![qlearning](./assets/qlearning.png)





### Comparison

Q-Learning learns the optimal policy, which moves along the cliff, but random exploration leads to higher chance of falling off. Thus, Q-Learning is less stable and has higher penalties. 

Meanwhile, SARSA finds a safer, but not the optimal path, which is further from the cliff.
