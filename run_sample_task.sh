cd ./booster_deploy
# gdk backend because the command doesnt work on wayland without it for some reason
GDK_BACKEND=x11 uv run python scripts/deploy.py --task k1_fight --mujoco