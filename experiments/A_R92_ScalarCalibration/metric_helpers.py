"""R89 reference metric definitions, also used by the R91 CPU evaluator."""
import numpy as np
import torch
from PIL import Image


def image(path):
    with Image.open(path) as im:
        return im.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom.item() > 0 else None


def clip_features(model, processor, imgs, texts, device):
    ip = processor(images=imgs, return_tensors='pt').to(device)
    tp = processor(text=texts, padding=True, truncation=True, return_tensors='pt').to(device)
    return model.get_image_features(**ip).float(), model.get_text_features(**tp).float()


def load_lpips(calibration_path, trunk_path, device):
    # Explicit local loads: LPIPS must not fetch trunk or calibration weights.
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', pnet_rand=True,
                         pretrained=False, eval_mode=True, verbose=False)
    trunk = torch.load(trunk_path, map_location='cpu', weights_only=True)
    calibration = torch.load(calibration_path, map_location='cpu', weights_only=True)
    mapped = {key: trunk['features.' + key.split('.', 1)[1]]
              for key in metric.net.state_dict()}
    metric.net.load_state_dict(mapped, strict=True)
    for i in range(5):
        prefix = 'lin'+str(i)+'.'
        weights = {key[len(prefix):]: value for key, value in calibration.items() if key.startswith(prefix)}
        getattr(metric, 'lin'+str(i)).load_state_dict(weights, strict=True)
    return metric.to(device).eval()


def lpips_tensor(im, device):
    array = np.asarray(im, dtype=np.float32).copy() / 127.5 - 1.
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
