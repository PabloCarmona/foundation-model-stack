from fms.models import get_model, list_models, list_variants
from torchinfo import summary
import os
import torch

models = list_models()
print(models)

variants = list_variants("mistral")
print(variants)

model = get_model(
  architecture="mistral",
  variant="7b",
  model_path=None,
  source=None,
  device_type="cpu",
  data_type=None,
  distributed_strategy=None,
  checkpoint_sharding=None,
  group=None,
  fused_weights=False,
  nlayers=1,
)

summary(model)

local_dir = "/Users/pxg/tiny-models/tiny-mistral-7b"
os.makedirs(local_dir, exist_ok=True)
model_file = os.path.join(local_dir, "model_state_dict.pt")
torch.save(model.state_dict(), model_file)
print("Model downloaded to:", model_file)
