set -x
./scripts/down.sh

TIMESTAMP=$(date +%s)
LOGS_COUNT=$(ls out/logs | wc -l)
FILES_COUNT=$(ls out/files | wc -l)

./scripts/chown.sh

if [ $LOGS_COUNT -ne 0 ];
then
    mkdir out/old-logs/logs-$TIMESTAMP
    cp out/logs/* out/old-logs/logs-$TIMESTAMP/
fi

if [ $FILES_COUNT -ne 0 ];
then
    mkdir out/old-files/files-$TIMESTAMP
    cp out/files/* out/old-files/files-$TIMESTAMP/
fi

./scripts/clear.sh
./scripts/up.sh
./scripts/logs.sh
./scripts/tail.sh