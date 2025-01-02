import numpy as np
import torch
# provides preprocessing for the environment.
class Preprocessor:

    # we normalize the grid to [0, 1].
    def preprocess(state:np.ndarray) -> torch.Tensor:
        return torch.tensor(state / np.max(state), dtype=torch.float32)
    
    # extracts heights, holes and bumpiness as feature metrics
    def extract_features(state: np.ndarray) -> np.ndarray:
        heights = np.max((state > 0) *  np.arange(state.shape[0])[:, None], axis=0)
        holes = np.sum((state == 0) & (np.cumsum(state > 0, axis=0) > 0))
        bumpiness = np.sum(np.abs(np.diff(heights)))
        return np.array([np.sum(heights), holes, bumpiness])