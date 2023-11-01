from typing import Dict, Hashable, List
try:
    import numpy as np
    import numpy.typing as npt
except ImportError as e:
    class FormatEnforcerAnalyzer: # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def report_raw_logits(self, *args, **kwargs):
            pass
        def generate_report_dict(self, *args, **kwargs):
            return {}
    raise ImportError('FormatEnforcerAnalyzer not available because numpy is not installed. Please install it with "pip install numpy"') from e

from . import TokenEnforcer

class FormatEnforcerAnalyzer:
    """A helper class to help analyze the format enforcer's behavior."""
    def __init__(self, token_enforcer: TokenEnforcer):
        self.token_enforcer = token_enforcer
        self.raw_logits: Dict[Hashable, npt.ArrayLike] = {}

    def report_raw_logits(self, output_tokens: List[int], logits: npt.ArrayLike):
        """Report what logits were generated for a specific token sequence. The logits must be before any processing / filtering."""
        self.raw_logits[tuple(output_tokens)] = logits

    def generate_report_dict(self, output_tokens: List[int]) -> dict:
        """Generate a report dict containing the analysis results for a specific output token sequence."""
        scores_matrix: List[npt.ArrayLike] = []
        allowed_tokens_matrix: List[List[int]] = []
        for idx in range(len(output_tokens)):
            prefix = output_tokens[:idx]
            prefix_tuple = tuple(prefix)
            if prefix_tuple in self.raw_logits:
                scores_matrix.append(self.raw_logits[prefix_tuple])
                allowed_tokens_matrix.append(self.token_enforcer.get_allowed_tokens(prefix))

        logits = np.array(scores_matrix) # n_tokens * vocab_size
        softmax_logits = _softmax(logits) # n_tokens * vocab_size
        original_indices = softmax_logits.argmax(axis=1) # n_tokens
        original_scores = _select_array(softmax_logits, original_indices) # n_tokens
        
        single_token_dict: Dict[int, str] = dict(self.token_enforcer.regular_tokens)
        def single_token_decoder(token_id: int) -> str:
            if token_id in single_token_dict:
                return single_token_dict[token_id]
            return self.token_enforcer.decoder([token_id])
        
        original_tokens = [single_token_decoder(idx) for idx in original_indices]
        
        penalty_matrix = np.full_like(softmax_logits, -np.inf)
        for row in range(penalty_matrix.shape[0]):
            penalty_matrix[row][allowed_tokens_matrix[row]] = 0
        enfored_softmax_logits = softmax_logits + penalty_matrix

        enforced_indices = enfored_softmax_logits.argmax(axis=1)
        enforced_scores = _select_array(enfored_softmax_logits, enforced_indices)

        enforced_tokens = [single_token_decoder(idx) for idx in enforced_indices]
        df_dict = {}  # In order to minimize the package's dependencies, we don't create a dataframe, but create a dataframe-like dictionary instead.
        df_dict['generated_token'] = enforced_tokens
        df_dict['generated_token_idx'] = enforced_indices.tolist()
        df_dict['generated_score'] = enforced_scores.tolist()
        df_dict['leading_token'] = original_tokens
        df_dict['leading_token_idx'] = original_indices.tolist()
        df_dict['leading_score'] = original_scores.tolist()

        return df_dict
    
def _softmax(arr: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in arr."""
    e_arr = np.exp(arr)
    return e_arr / np.sum(e_arr, axis=1, keepdims=True)

def _select_array(arr: np.ndarray, index_array: np.ndarray) -> np.ndarray:
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    return np.take_along_axis(arr, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)