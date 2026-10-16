"""Fixture-measurable state plus robot-to-handle relative geometry."""
import numpy as np

EXTRA_FEATURES = ['target_angle', 'handle_minus_eef_x', 'handle_minus_eef_y',
                 'handle_minus_eef_z', 'eef_to_handle_qw', 'eef_to_handle_qx',
                 'eef_to_handle_qy', 'eef_to_handle_qz']


def relative_features(raw):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != 39:
        raise ValueError('Expected original robot26 + environment13')
    # wxyz; canonical quaternion sign prevents equivalent q/-q discontinuity.
    q = raw[:, 21:25].copy()
    h = raw[:, 35:39].copy()
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-8)
    w = np.sum(q * h, axis=-1, keepdims=True)
    xyz = q[:, :1] * h[:, 1:] - h[:, :1] * q[:, 1:] - np.cross(q[:, 1:], h[:, 1:])
    relq = np.concatenate((w, xyz), -1)
    relq = np.where(relq[:, :1] < 0, -relq, relq)
    target = np.ones((len(raw), 1), np.float32)
    return np.concatenate((target, raw[:, 32:35] - raw[:, 18:21], relq), -1)


def expand(raw):
    return np.concatenate((np.asarray(raw, np.float32), relative_features(raw)), -1)


class RecoveryAdapter:
    def __init__(self, mean, std, length=200):
        import sentencepiece
        from pi05.state_adapter import TOKENIZER
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        if self.mean.shape != (47,) or self.std.shape != (47,) or np.any(self.std <= 0):
            raise ValueError('Invalid recovery normalization')
        self.length = length
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))

    def encode(self, raw):
        if raw.ndim != 2 or raw.shape[1] != 47 or not np.isfinite(raw).all():
            raise ValueError('Invalid recovery features')
        z = (raw - self.mean) / self.std
        bins = np.digitize(2 / np.pi * np.arctan(z), np.linspace(-1, 1, 257)[:-1]) - 1
        tokens = np.zeros((len(raw), self.length), np.int64)
        masks = np.zeros_like(tokens, dtype=bool)
        for i, row in enumerate(bins):
            text = 'Task: open the door, State: ' + ' '.join(map(str, row)) + ';\nAction: '
            ids = self.sp.encode(text, add_bos=True)
            if len(ids) > self.length:
                raise RuntimeError(f'Recovery token truncation: {len(ids)}')
            tokens[i, :len(ids)] = ids
            masks[i, :len(ids)] = True
        return tokens, masks
