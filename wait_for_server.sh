#!/bin/bash
# Wait for GCP server to be ready

EXTERNAL_IP="34.72.141.111"
MAX_ATTEMPTS=30
ATTEMPT=1

echo "Waiting for server at http://$EXTERNAL_IP:8080 to start..."
echo "(This may take 2-3 minutes for first-time setup)"
echo ""

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo -n "Attempt $ATTEMPT/$MAX_ATTEMPTS: "

    if curl -s --max-time 5 "http://$EXTERNAL_IP:8080/search?query=test" >/dev/null 2>&1; then
        echo "Server is UP!"
        break
    else
        echo "Waiting..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

echo ""

if [ $ATTEMPT -le $MAX_ATTEMPTS ]; then
    echo "========================================="
    echo "Server is ready!"
    echo "========================================="
    echo ""
    echo "Testing with Mount Everest query..."
    echo ""
    curl -s "http://$EXTERNAL_IP:8080/search?query=Mount+Everest" | python3 -m json.tool | head -30
    echo ""
    echo "========================================="
    echo "Success! Server is responding."
    echo "========================================="
else
    echo "Server didn't respond in time."
    echo ""
    echo "Check logs with:"
    echo "  gcloud compute ssh ir-search-engine --zone=us-central1-a --command 'sudo journalctl -u ir-search -n 100'"
fi
