# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## VALUE ITERATION ALGORITHM
### Step 1: Set the value of each state to 0 (initial guess).
### Step 2: Look at all the actions you can take from that state (like moving up, down, left, or right).
### Step 3: Calculate the expected value of each action (i.e., how good that action is based on its possible results).
### Step 4: Pick the action that gives the highest value and update the value of the state with that number.
### Step 5: Keep updating the values for all states until the difference between the old and new values is very small.
### Step 6: Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state.

## VALUE ITERATION FUNCTION
### Name: SANTHANA LAKSHMI K
### Register Number: 212222240091
Include the value iteration function

```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q= np.zeros((len(P), len(P[0])), dtype=np.float64)
      for s in range((len(P))):
        for a in range(len(P[s])):
          for prob, next_state, reward, done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
        break
      V= np.max(Q, axis=1)
      pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=1))} [s]
    return V, pi
```
### Frozen Lake environment
```
envdesc  = ['SFFF','FHFH','FFFH', 'GFFF']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 12
P = env.env.P
```



## OUTPUT:
### Optimal policy :
![image](https://github.com/user-attachments/assets/2f204bba-ebed-4c21-8552-4934c4416433)

### Goal
![image](https://github.com/user-attachments/assets/769f603f-9705-4811-bca5-fe387130cdc0)

### State value function

![image](https://github.com/user-attachments/assets/5f72d040-397b-4cee-abb7-fb165f68fc5b)



## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm
