import abc

import torch
import math


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """
        pass

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """
        pass


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        #raise NotImplementedError()
        #Some of the code has been written with help of Gemini
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding layer
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Positional embedding (optional but helpful)
        self.use_pos_embedding = True
        if self.use_pos_embedding:
            # Maximum sequence length for 100x150 image with patch_size=5: 20*30=600
            max_seq_len = 1000
            self.pos_embedding = torch.nn.Embedding(max_seq_len, d_latent)
        
        # Transformer layers
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=8,
                dim_feedforward=d_latent * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(6)  # 6 transformer layers
        ])
        
        # Output projection to vocabulary
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        # Initialize embeddings
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.use_pos_embedding:
            torch.nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Initialize output projection
        torch.nn.init.normal_(self.output_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.output_proj.bias)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #raise NotImplementedError()
        B, h, w = x.shape
        
        # Flatten to sequence: (B, h, w) -> (B, h*w)
        x_flat = x.view(B, h * w)  # (B, seq_len)
        seq_len = h * w
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embed tokens
        token_emb = self.token_embedding(x_flat)  # (B, seq_len, d_latent)
        
        # Add positional embeddings if enabled
        if self.use_pos_embedding:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
            embeddings = token_emb + pos_emb
        else:
            embeddings = token_emb
        
        # Shift embeddings by 1 position for autoregressive prediction
        # The model should predict token i+1 given tokens 0...i
        shifted_embeddings = torch.zeros_like(embeddings)
        shifted_embeddings[:, 1:] = embeddings[:, :-1]  # Shift right by 1
        # First position gets zero embedding (predicts first token from nothing)
        
        # Pass through transformer layers
        hidden = shifted_embeddings
        for layer in self.transformer_layers:
            hidden = layer(hidden, src_mask=causal_mask)  # (B, seq_len, d_latent)
        
        # Project to vocabulary
        logits = self.output_proj(hidden)  # (B, seq_len, n_tokens)
        
        # Reshape back to image format
        logits = logits.view(B, h, w, self.n_tokens)  # (B, h, w, n_tokens)
        
        # No additional losses for basic autoregressive model
        additional_losses = {}
        
        return logits, additional_losses

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        #raise NotImplementedError()
         if device is None:
            device = next(self.parameters()).device
        
         self.eval()
        
         # Initialize with zeros (or random tokens)
         generated = torch.zeros(B, h, w, dtype=torch.long, device=device)
        
         with torch.no_grad():
            # Generate tokens one by one in raster order
            for i in range(h):
                for j in range(w):
                    # Get logits for current position
                    logits, _ = self.forward(generated)  # (B, h, w, n_tokens)
                    
                    # Sample from the distribution at position (i, j)
                    probs = torch.softmax(logits[:, i, j], dim=-1)  # (B, n_tokens)
                    next_token = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
                    
                    # Update the generated sequence
                    generated[:, i, j] = next_token
        
         return generated
