# Kinich Quantum — Hybrid Quantum-Classical Compute

## Project Identity
- **Repo**: `BelizeChain/kinich-quantum`
- **Role**: Hybrid quantum-classical computing orchestration for BelizeChain
- **Language**: Python
- **Branch**: `main` (default)

## Features
- Multi-backend quantum execution (Qiskit, Cirq, PennyLane)
- Quantum error mitigation and circuit optimization
- BelizeChain pallet integration (quantum randomness, quantum proofs)
- Production monitoring and health checks
- REST API server (`api_server.py`)

## Azure Deployment Target
- **ACR**: `belizechainacr.azurecr.io` → image: `belizechainacr.azurecr.io/kinich`
- **AKS**: `belizechain-aks` (Free tier, 1x Standard_D2s_v3, K8s v1.33.6)
- **Resource Group**: `BelizeChain` in `centralus`
- **Subscription**: `77e6d0a2-78d2-4568-9f5a-34bd62357c40`
- **Tenant**: `belizechain.org`

## Deployment Status: Phase 2 — TODO
### What needs to be done:
1. **Verify Dockerfile** — Ensure Python dependencies install cleanly, API server starts
2. **Update deploy.yml** — Migrate from VM/SSH to AKS deployment:
   - Use `azure/login@v2` with `${{ secrets.AZURE_CREDENTIALS }}`
   - Use `azure/aks-set-context@v4` with `${{ secrets.AKS_CLUSTER_NAME }}`
   - Push image to `belizechainacr.azurecr.io/kinich`
   - `kubectl apply` Deployment + Service (expose port 8000)
3. **Configure GitHub Secrets** (copy from belizechain repo or set independently):
   - `ACR_USERNAME` = `belizechainacr`
   - `ACR_PASSWORD` = (get from `az acr credential show --name belizechainacr`)
   - `AZURE_CREDENTIALS` = (service principal JSON)
   - `AZURE_RESOURCE_GROUP` = `BelizeChain`
   - `AKS_CLUSTER_NAME` = `belizechain-aks`
4. **K8s namespace**: Deploy into `belizechain` namespace
5. **Resource limits**: CPU-only mode (no GPU on Free tier), limit to 100m-500m CPU, 256Mi-1Gi RAM
6. **Connect to blockchain**: Configure connection to `ws://belizechain-node.belizechain.svc.cluster.local:9944`

## Sibling Services (same AKS cluster)
| Service | Image | Ports |
|---------|-------|-------|
| belizechain-node | `belizechainacr.azurecr.io/belizechain-node` | 30333, 9944, 9615 |
| ui | `belizechainacr.azurecr.io/ui` | 80 |
| nawal-ai | `belizechainacr.azurecr.io/nawal` | 8001 |
| pakit-storage | `belizechainacr.azurecr.io/pakit` | 8002 |

## Dev Commands
```bash
pip install -r requirements.txt          # Install dependencies
python api_server.py                     # Run API server
docker build -t belizechainacr.azurecr.io/kinich .  # Docker image
pytest                                   # Run tests
```

## Constraints
- **CPU-only**: No GPU available on AKS Free tier — quantum simulation must use CPU backends
- **Shared node**: All services share 2 vCPU / 8GB RAM — keep resource requests minimal
- **Cost ceiling**: ~$75/mo total for ALL services
