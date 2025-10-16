import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """
        pass

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """
        pass


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        #raise NotImplementedError()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        
        # Better initialization for BSQ layers
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim, bias=False)
        
        # Initialize with smaller weights for better training
        torch.nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        #raise NotImplementedError()
        # Down-project to codebook dimension
        x = self.down_proj(x)  # (..., codebook_bits)
        
        # L2 normalization along the last dimension
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # Apply differentiable sign function
        x = diff_sign(x)
        
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        #raise NotImplementedError()
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=5, latent_dim=128, bottleneck=128)
        #raise NotImplementedError()
        #Some of the code has been wrirren with help of Gemeni
        self.patch_size = 5
        self.latent_dim = 128
        self.codebook_bits = 10
        
        # BSQ quantizer
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=self.bottleneck)
        
        # Add a commitment loss weight for better training
        self.commitment_weight = 0.1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().encode(x)
        quantized = self.bsq.encode(latent)
        return quantized

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.bsq.decode(x)
        image = super().decode(latent)
        return image

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().encode(x)
        indices = self.bsq.encode_index(latent)
        return indices

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.bsq.decode_index(x)
        image = super().decode(latent)
        return image

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with commitment loss for better training."""
        # Encode to latent space
        latent = super().encode(x)  # (B, h, w, bottleneck)
        
        # BSQ encode and decode
        quantized_codes = self.bsq.encode(latent)  # (B, h, w, codebook_bits)
        quantized_latent = self.bsq.decode(quantized_codes)  # (B, h, w, bottleneck)
        
        # Decode back to image
        x_reconstructed = super().decode(quantized_latent)  # (B, H, W, 3)
        
        # Commitment loss to encourage encoder to commit to codebook
        commitment_loss = torch.nn.functional.mse_loss(latent, quantized_latent.detach())
        
        # Monitor codebook usage
        indices = self.bsq.encode_index(latent)
        cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
        
        additional_losses = {
            "commitment": commitment_loss * self.commitment_weight,
            "cb0": (cnt == 0).float().mean().detach(),
            "cb1": (cnt == 1).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
            "cb_max": cnt.max().float().detach(),
            "cb_mean": cnt.float().mean().detach(),
            "cb_unique": (cnt > 0).float().sum().detach(),
        }
        
        return x_reconstructed, additional_losses
