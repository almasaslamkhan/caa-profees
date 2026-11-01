import os

os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",.oci.oraclecloud.com"
