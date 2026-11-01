#! /bin/bash

# Define the server IP and port
SERVER_IP="0.0.0.0"
PORT=${PORT_PROFEES:-8002}

# Function to check if a port is in use
is_port_in_use() {
    nc -z -v -w5 $SERVER_IP $PORT &> /dev/null
    return $?
}

# Function to check the health status of the server (customize this URL as per your health check endpoint)
check_server_health() {
    local url="http://$SERVER_IP:$PORT/health"
    curl --silent --fail $url > /dev/null
    return $?
}

# Step 1: Check if the port is in use
if is_port_in_use; then
    echo "Port $PORT is already in use. Attempting to find and kill the process..."

    # Find and kill the process occupying the port
    PID=$(lsof -t -i:$PORT)
    if [ -n "$PID" ]; then
        echo "Killing process with PID: $PID"
        kill -9 $PID
        sleep 2
    else
        echo "No process found using port $PORT"
    fi
else
    echo "Port $PORT is free. Proceeding with server startup..."
fi

# ToDo: once VDB is done, should have it as of the codebase and include it automatically, instead of a full path.
export LOG_LEVEL=${LOG_LEVEL_PROFEES:-"INFO"}

# # Start the application - for 26.05 data contract
# python cdrc/profees_coding/deploy/api/model/api.py \
#     --log_level "${LOG_LEVEL}" \
#     --port "${PORT}" \
#     > /dev/null 2>&1

# Start the application - for 25.10 data contract
python cdrc/profees_coding/deploy/api/model/api.py \
    --log_level ${LOG_LEVEL}\
    --port ${PORT} /dev/null 2>&1