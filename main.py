from reacher.train import make_plot,setup,train
import os
from unityagents import UnityEnvironment


env = UnityEnvironment('Reacher.app')

if __name__ == "__main__":
    for name in ['score.png','scores.npz','checkpoint.pth']:
        if os.path.isfile(name):
            os.remove(name)

    agent = setup(env)
    print("Training the agent.")
    train(agent=agent,env=env)

    print('Make training plot')
    make_plot()