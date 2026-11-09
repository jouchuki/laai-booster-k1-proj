"""Train the Booster K1 walking policy with mjlab.

  python train_k1.py Mjlab-Velocity-Flat-Booster-K1 --env.scene.num-envs 4096
"""
import k1_task  # noqa: F401  (registers the K1 task)
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
