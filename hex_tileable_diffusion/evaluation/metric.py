import torch, numpy as np
from PIL import Image
from torchvision import transforms
import piq
import lpips
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from textile import Textile


class Metrics:

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._textile: Textile | None = None
        self._lpips_net: lpips.LPIPS | None = None
        self._incept: InceptionV3 | None = None

    def textile(self, gen) -> float:
        if self._textile is None:
            self._textile = Textile()
        if isinstance(gen, Image.Image):
            gen = transforms.ToTensor()(gen.convert("RGB")).unsqueeze(0)
        elif isinstance(gen, np.ndarray):
            gen = torch.from_numpy(gen).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return float(self._textile(gen.to(self.device)).item())

    def ssim(self, ref: torch.Tensor, gen: torch.Tensor) -> float:
        # piq.ssim expects [0,1] tensors
        return float(piq.ssim(ref, gen, data_range=1.0))

    def lpips(self, ref: torch.Tensor, gen: torch.Tensor) -> float:
        if self._lpips_net is None:
            self._lpips_net = lpips.LPIPS(net="alex").to(self.device).eval()
        # lpips expects [-1,1]
        r, g = (ref * 2 - 1).to(self.device), (gen * 2 - 1).to(self.device)
        with torch.no_grad():
            return float(self._lpips_net(r, g).item())

    def si_fid(self, ref: torch.Tensor, gen: torch.Tensor, eps: float = 1e-6) -> float:
        if self._incept is None:
            self._incept = InceptionV3([2]).to(self.device).eval()

        def feats(x: torch.Tensor) -> np.ndarray:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                f = self._incept(x.to(self.device))[0]  # [B, C, H, W]
            b, c, h, w = f.shape
            return f.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu().numpy()

        f1, f2 = feats(ref), feats(gen)
        mu1, mu2 = f1.mean(0), f2.mean(0)
        s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        reg = eps * np.eye(s1.shape[0])
        covmean = linalg.sqrtm((s1 + reg) @ (s2 + reg), disp=False)[0]
        if np.iscomplexobj(covmean): covmean = covmean.real
        diff = mu1 - mu2
        return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
