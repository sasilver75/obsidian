---
aliases:
  - MCTS
---
MCTS is a probabilistic algorithm using random simulations to guide the search for a best move in a game. 

Game trees are graphs representing all possible game states within a combinatorial game.
- Nodes represent particular states of a game
- Edges represent possible next states from a given state.
- The leaf nodes of the tree represent an ending condition for the game (eg a win, loss, tie).

There are various algorithms that we could use to solve this problem, including:
- Minimax
- Minimax with alpha, beta pruning
- A* Search

These algorithms traverse the complete game tree to pick the optimal move -- but what if the game size is very large, with some high branching factor? It might not be computationally feasible to traverse the entire game to pick the best move!
- It's in this case that [[Monte-Carlo Tree Search|MCTS]] is very effective.

MCTS uses a heuristic search algorithm to solve the game tree!
Unlike traditional algorithms that rely upon exhaustive exploration of the entire seek area, MCTS specializes in sampling and exploring only promising areas of the hunt area.
- The central idea is that we build a seek tree incrementally, by simulating random rollouts from the current state until a terminal state is reached; In each playout, the final game result of each playout is then used to weight the nodes in the game tree so that better nodes are more likely to be chosen in future playouts.
- As the search continues, MCTS dynamically balances exploration and exploitation, considering moves by prioritizing both choices with high win ratios and by exploring unexplored/less-explored moves. We use Upper Confidence Bounds (UCB) to help us select moves. 

It has four phases:
1. ==Selection==: We start from the root $R$, and select successive child nodes until a leaf node $L$ is reached. The root is the current game state, and a leaf is any node that has a potential child from which no simulation (playout) has yet been initiated.
	- The algorithm uses the following Upper Confidence Bound (UCB) formula to calculate the state value of all possible next moves, and picks the one which gives the maximum value.
	- ![[Pasted image 20240616212503.png|250]]
	- v_i is the exploitation term; the second term lg(N/n_i) is the exploitation term. This defines the weightage between exploitation of known, high-valued nodes, and relatively unexplored nodes.
	- At the beginning, when no node is explored, we make a random selection, because there's no data available for a more education selection. When a node is unexplored, the second term becomes $\infty$ , and thus obtains a maximum possible value, and automatically becomes a candidate for selection; thus, the equation makes sure all children get selected at least once.
2. ==Expansion==
	- Unless the leaf node reached above ends the game decisively, create one (or more) child nodes and choose a node $C$ from one of them. Child nodes are any valid moves from the game position defined by our leaf node $L$.
3. ==Simulation==: Complete a random rollout from our node $C$. This step is sometimes also also called playout/rollout. A playout may be as simple as choosing uniform, random moves until the game is decided.
4. ==Backpropagation==: Use the result of the playout to update information in he nodes on the path from $C$ to $R$.

![[Pasted image 20240616214707.png]]
Above: Each node shows the ratio of wins to total plays from that point in the game tree.
Rounds of search are repeated as long as the time allotted to a move remains; the move with the most simulations made (i.e. the highest denominator) is chosen as the final answer ((?)).

The main difficulty in selecting child nodes is maintaining some balance between *exploitation* of moves with high-average-win-rates, and *exploration* of moves with few simulations.
- The first formula for balancing this was called Upper confidence Bound, UCB (or UCT (UCB for Trees)).
- Modern implementations of MCTS are based on some variant of UCT.

Optimizations
- MCTS can use either *light* or *heavy* playouts, where light playouts consist of random moves, while heavy playouts/rollouts apply various heuristics to influence the choice of moves (interestingly, playing suboptimally in simulations sometimes makes a MCTS program play stronger overall
- Basic MCTS exploratory phase can be shortened in certain classes of games using a technique called RAVE (Rapid Action Value Estimation)
- MCTS can be concurrently executed by many threads or processes.