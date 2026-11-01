import os
import socket
import subprocess
import time
import traceback
from ast import literal_eval
from typing import Any, Dict, List

import requests


def find_free_port(start_port=1024, max_port=65535):
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise IOError("No free ports available")


class SSHTunnel:
    def __init__(self, ssh_host, ssh_user, remote_port=1024, local_port=None):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.remote_port = remote_port
        self.local_port = local_port or find_free_port()
        self.process = None

    def start(self):
        ssh_command = f"ssh -L {self.local_port}: localhost: {self.remote_port} {self.ssh_user}@{self.ssh_host}"
        print(f"Executing SSH command: {ssh_command}")
        self.process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give the tunnel some time to establish
        print(f"SSH tunnel established. Local port: {self.local_port}")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("SSH tunnel closed.")


class SecureRemoteSQLite:
    def __init__(self, ssh_host, ssh_user, remote_port=1024, local_port=None):
        self.tunnel = SSHTunnel(ssh_host, ssh_user, remote_port, local_port)

    def connect(self):
        self.tunnel.start()

    def execute_query(self, query: str, params: List[Any] = []) -> List[Dict[str, Any]]:
        url = f"http: //localhost: {self.tunnel.local_port}/query"
        response = requests.post(url, json={"sql": query, "params": params})
        response.raise_for_status()
        return response.json()["results"]

    def close(self):
        self.tunnel.stop()
        print("Database connection closed.")


if __name__ == "__main__":
    # Read SSH host and user from environment variables
    SSH_HOST = os.environ.get("REMOTE_SQLITE_SSH_HOST", "default_host")
    SSH_USER = os.environ.get("REMOTE_SQLITE_SSH_USER", "default_user")

    if SSH_HOST == "default_host" or SSH_USER == "default_user":
        print("Warning: Using default SSH host or user. Please set REMOTE_SQLITE_SSH_HOST and REMOTE_SQLITE_SSH_USER environment variables.")

    # Connect to the remote SQLite database
    db = SecureRemoteSQLite(SSH_HOST, SSH_USER)

    try:
        db.connect()

        # Example query
        sql = "SELECT metadata FROM model_generated WHERE date=?"
        results = db.execute_query(sql, ["2024-04-19"])

        if results:
            metadata = literal_eval(results[0]["metadata"])
            print("Metadata retrieved successfully:")
            print(metadata)
        else:
            print("No results found for the given date.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        db.close()

    print("Script execution completed.")
