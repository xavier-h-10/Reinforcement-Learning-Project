from gym.wrappers import Monitor
import torch

# be aware that gym<=0.12.0, gym has removed monitor wrapper in higher version

def export_video(env, agent, max_episode=1000):
    assert env, agent is not None
    env = Monitor(env, './result', force=True)
    agent.actor.load_state_dict(torch.load('ddpg_actor.pth'))
    agent.critic.load_state_dict(torch.load('ddpg_critic.pth'))

    state = env.reset()
    for t in range(max_episode):
        action = agent.choose_action(state, use_noise=False)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()
