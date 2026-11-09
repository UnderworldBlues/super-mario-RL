from wrapper import make_env

env = make_env()
obs = env.reset()
print(f"formato do estado inicial: {obs.shape} (Esperado 4 frames de 84x84)")
print(f"total de ações disponiveis: {env.action_space.n} (esperado 12)")

pronto = False
step_count = 0
while not pronto and step_count < 100:
    action = env.action_space.sample()
    obs, reward, pronto, info = env.step(action)
    print(f"Passo: {step_count:03d} | Ação: {action:02d} | Recompensa: {reward:5.1f} | Morte/Fim: {pronto}")
    step_count += 1

env.close()
print("\nteste feito")