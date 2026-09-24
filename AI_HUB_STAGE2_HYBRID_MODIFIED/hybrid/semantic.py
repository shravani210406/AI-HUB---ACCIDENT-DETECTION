"""Frozen normalized CLIP visual embeddings plus cached text similarities."""
import numpy as np
from .config import resolve

PROMPTS = ['normal road traffic','vehicles driving normally','traffic congestion',
           'cars stopped at traffic signal','vehicle collision','road traffic accident',
           'motorcycle accident','vehicle overturned','vehicles impacting each other',
           'damaged vehicles after collision']

class SceneEncoder:
    def __init__(self,c):
        import torch
        import open_clip
        torch.set_num_threads(c['torch_threads'])
        self.torch = torch
        pretrained=str(resolve(c['clip_weights'])) if c.get('clip_weights') else c['clip_pretrained']
        self.model,_,self.preprocess = open_clip.create_model_and_transforms(
            c['clip_model'],pretrained=pretrained,cache_dir=str(resolve('models/clip_cache')))
        self.model.eval()
        with torch.inference_mode():
            tokens = open_clip.get_tokenizer(c['clip_model'])(PROMPTS)
            self.text = self.model.encode_text(tokens,normalize=True).cpu().numpy()
        self.compiled = None
        if c.get('clip_openvino'):
            from .backends import compile_openvino
            self.compiled = compile_openvino(c['clip_openvino'],c)

    def encode(self,frame):
        from PIL import Image
        x = self.preprocess(Image.fromarray(frame[:,:,::-1])).unsqueeze(0)
        if self.compiled is not None:
            e = self.compiled([x.numpy()])[self.compiled.output(0)][0]
            e = e / max(np.linalg.norm(e),1e-8)
        else:
            with self.torch.inference_mode():
                e = self.model.encode_image(x,normalize=True)[0].cpu().numpy()
        return np.concatenate([e,self.text@e]).astype(np.float32)
