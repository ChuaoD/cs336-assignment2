# Section 7: FSDP Accounting

| Component | Section 6 optimizer sharding | FSDP on two GPUs | Memory saved per GPU |
|---|---:|---:|---:|
| FP32 weights | $4P = 13.6272\ \mathrm{GB}$ | $4P/2 = 6.8136\ \mathrm{GB}$ | $6.8136\ \mathrm{GB}$ |
| FP32 gradients | $4P = 13.6272\ \mathrm{GB}$ | $4P/2 = 6.8136\ \mathrm{GB}$ | $6.8136\ \mathrm{GB}$ |
| AdamW moments | Already sharded | Still sharded | Approximately $0\ \mathrm{GB}$ additional |
| Activations and small buffers | Unchanged | Unchanged | $0\ \mathrm{GB}$ |
| **Total additional saving** |  |  | **$13.6272\ \mathrm{GB}$** |

With two GPUs, FSDP halves the storage for the FP32 weights and gradients, saving approximately $13.6272\ \mathrm{GB}$ in addition to the optimizer-state sharding from Section 6; the AdamW states are already sharded, while activations and small replicated buffers are unchanged. Starting from the maximum measured optimizer-sharded peak of about $41.457\ \mathrm{GB}$, this gives an expected FSDP peak of approximately $27.83\ \mathrm{GB}$ when all-gather buffers and the negligible replicated norm weights are ignored.
