#!/bin/bash
# Determine MODE_PROFEES unless it's explicitly set to "MOCK"
if [[ -z "$MODE_PROFEES" || "$MODE_PROFEES" != "MOCK" ]]; then
    if [[ -z "$CLUSTER_NAME" ]]; then
        MODE_PROFEES="LOCAL"
        echo "ℹ️  CLUSTER_NAME is not set. Defaulting MODE_PROFEES to LOCAL."
    else
        echo "CLUSTER_NAME = $CLUSTER_NAME"

        # Extract PREFIX and REGION, convert to lowercase
        PREFIX="$(echo "${CLUSTER_NAME%%-*}" | tr '[:upper:]' '[:lower:]')"

        # Set MODE_PROFEES based on PREFIX and REGION
        if [[ "$PREFIX" == "pc" ]]; then
            MODE_PROFEES="PROD"
        else
            MODE_PROFEES="DEV"
        fi
    fi
fi

export MODE_PROFEES
echo "✅ MODE_PROFEES is set to: $MODE_PROFEES"
echo "🚀 Running entrypoint with MODE = $MODE_PROFEES"
# -------------------- Determine and Run Entrypoint --------------------
ENTRYPOINT_PATH="cdrc/profees_coding/deploy/api/model/runpoint.sh"

# -------------------- Keep-Alive Pings to Prevent Cold Start --------------------
keep_alive_profees() {
    local base_url="http://oda-genai-proxy-service-java.max:8421/openai/deployments"
    local api_version="2024-02-01"
    local headers=(
        -H "Connection: keep-alive"
        -H "Content-Type: application/json"
        -H "api-key: EMPTY"
        -H "Service-Name: profees"
    )
    
    while true; do
        echo "⏱  Sending keep-alive requests for $MODE_PROFEES..."

        if [[ "$MODE_PROFEES" == "PROD" ]]; then
            curl --silent --location "${base_url}/caa-gpt-4.1-2025-04-14-prod/chat/completions?api-version=${api_version}" \
                "${headers[@]}" \
                --data '{
                  "messages": [
                    {"role": "system", "content": "You are a medical doctor."},
                    {"role": "user", "content": "Give me one word"}
                  ],
                  "temperature": 0,
                  "stream": false
                }' > /dev/null

            curl --silent --location "${base_url}/caa-text-embedding-3-large-prod/embeddings?api-version=${api_version}" \
                "${headers[@]}" \
                --data '{"input": "Tell me a joke"}' > /dev/null

        elif [[ "$MODE_PROFEES" == "DEV" ]]; then
            curl --silent --location "${base_url}/caa-gpt-4.1-2025-04-14-dev/chat/completions?api-version=${api_version}" \
                "${headers[@]}" \
                --data '{
                  "messages": [
                    {"role": "system", "content": "You are a medical doctor."},
                    {"role": "user", "content": "Give me one word"}
                  ],
                  "temperature": 0,
                  "stream": false
                }' > /dev/null

            curl --silent --location "${base_url}/caa-text-embedding-3-large-dev/embeddings?api-version=${api_version}" \
                "${headers[@]}" \
                --data '{"input": "Tell me a joke"}' > /dev/null
        fi

        echo "✅ Keep-alive requests sent for $MODE_PROFEES. Sleeping for 1 hour..."
        sleep 3600
    done
}

if [[ "$MODE_PROFEES" == "PROD" || "$MODE_PROFEES" == "DEV" ]]; then
    keep_alive_profees &
    echo "🌀 Started keep-alive background job for $MODE_PROFEES mode"
fi

chmod +x "$ENTRYPOINT_PATH"
exec "$ENTRYPOINT_PATH"
