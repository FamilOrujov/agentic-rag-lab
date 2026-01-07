set -euo pipefail

# Starts FastAPI app under keploy and records traffic into ./keploy
# Uses keploy.yaml, so I don't have to pass -c/-p flags
keploy record
