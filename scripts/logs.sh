set -x
CONTAINER_ID=$(docker compose ps -q)
docker container logs -f "$CONTAINER_ID" >& out/logs/logs.txt &
LOGS_PID=$!
echo "Logs process ID: $LOGS_PID"