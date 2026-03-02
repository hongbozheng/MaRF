# LLaMA-Style Encoder for Retrieval

## Train Encoder
### Train Configuration
- To modify train configuration, check `cfgs/models/dual_enc.yaml` file
- To modify dataset configuration, check `cfgs/datasets/contrastive_expr.yaml` file

### Train
To train encoder
```
./train_model.py --cfgs cfgs/models/dual_enc.yaml --dataset cfgs/datasets/contrastive_expr.yaml
```
