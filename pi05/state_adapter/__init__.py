"""All39 measured features enter the pi05 discrete state prefix."""
from pathlib import Path
from runtime_paths import PI05_TOKENIZER
import numpy as np,sentencepiece

TOKENIZER=PI05_TOKENIZER
FEATURE_COUNT=39
PROMPT='open the door'

class StateAdapter:
    def __init__(self,mean,std,length=200):
        self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32)
        self.length=length;self.sp=sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
    def encode(self,raw):
        assert raw.ndim==2 and raw.shape[1]==FEATURE_COUNT
        z=(raw-self.mean)/self.std
        # Invertible bounded representation before OpenPI's intrinsic discrete
        # state bins. Avoid silently clipping standardized state to [-1,1].
        bounded=2/np.pi*np.arctan(z)
        bins=np.digitize(bounded,np.linspace(-1,1,257)[:-1])-1
        ids=np.zeros((len(raw),self.length),np.int64);masks=np.zeros_like(ids,dtype=bool)
        for i,row in enumerate(bins):
            text=f'Task: {PROMPT}, State: '+ ' '.join(map(str,row))+';\nAction: '
            tokens=self.sp.encode(text,add_bos=True)
            if len(tokens)>self.length:raise RuntimeError(f'GT token truncation: {len(tokens)}')
            ids[i,:len(tokens)]=tokens;masks[i,:len(tokens)]=True
        return ids,masks
