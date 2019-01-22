import hw3 
import numpy as np

observation,AgentPos,MonsterPos  = hw3.AtariEnv.reset()

print("Environment ",observation)
print("Agent Position",AgentPos)
print("Monster Pos ",MonsterPos)


action = np.random.randint(0,3,5)

for i in range(5):
	if action[i] == 0:
		print("Agent to move right ")
	elif action[i] == 1:
		print("Agent is move left")
	elif action[i] == 2:
		print("Agent to move up")
	elif action[i] == 3:
		print("Agent to move down")


	observation,AgentPos,MonsterPos = hw3.AtariEnv.step(observation,AgentPos,MonsterPos,action[i])



print(observation)