set -euo pipefail

docker stop ros2-booster-k1
docker rm ros2-booster-k1
docker image rm ros2-booster-image
docker build -t ros2-booster-image . && \
    docker run --name ros2-booster-k1 -d ros2-booster-image