try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

class TokenList:
    def __init__(self, use_bitmask: bool, vocab_size: int):
        self.use_bitmask = use_bitmask
        
        if use_bitmask and not _HAS_TORCH:
            raise ValueError("LMFormatEnforcer bitmasks require torch")
        
        if self.use_bitmask:
            tensor_size = (vocab_size + 31) // 32
            self.allowed_tokens = torch.zeros(tensor_size, dtype=torch.int32)
        else:
            self.allowed_tokens = []

    
    def append(self, token_id):
        if self.use_bitmask:
            element_index = token_id >> 5
            bit_index = token_id & 0x1f
            self.allowed_tokens[element_index] |= (1 << bit_index)
        else:
            self.allowed_tokens.append(token_id)

    def extend(self, token_ids):
        if self.use_bitmask:
            if isinstance(token_ids, torch.Tensor):
                torch.Tensor.bitwise_or_(self.allowed_tokens, token_ids)
            else:
                for token_id in token_ids:
                    element_index = token_id >> 5
                    bit_index = token_id & 0x1f
                    self.allowed_tokens[element_index] |= (1 << bit_index)
        else:
            self.allowed_tokens.extend(token_ids)

    def is_token_allowed(self, token_id) -> bool:
        if self.use_bitmask:
            element_index = token_id // 32
            bit_index = token_id % 32
            return (self.allowed_tokens[element_index] & (1 << bit_index)) != 0
        else:
            return token_id in self.allowed_tokens
