LAAI Humanoid Cycling project

### Dependencies

to install all dependencies, first install `uv` package manager

heres how (for Mac)

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then run this in the repo folder

```
uv sync
```

### Initial plan

1. Prove sim-to-real with walking. Train a standard walking policy for our robot and run it on the real one. This tests the whole pipeline (model, simulator, training, deployment), and walking is also the base skill for riding a balance bike.
2. Build the balance-bike model in simulation: frame, handlebar and fork, two wheels, saddle. No pedals.
3. Sit on a supported bike. The bike is held upright; the robot learns to sit stably with hands on the bars and feet on the ground.
4. Walk the bike, supported. Still held upright, the robot pushes off with alternating feet to roll forward.
5. Free bike, feet as support. Remove the support. The robot strides forward and can put a foot down to catch itself, while learning to balance by steering.
6. Simulation to real robot. Carefully test on real robot

extra things: Stride and glide. Lift both feet and coast, balancing by steering only, then push again, steering on command. Ride in a requested direction.