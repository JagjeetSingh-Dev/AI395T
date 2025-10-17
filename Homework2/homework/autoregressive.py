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
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        pass

class AutoregressiveModel(torch.nn.Module, Autoregressive):
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding layer
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Positional embedding
        self.use_pos_embedding = True
        if self.use_pos_embedding:
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
            for _ in range(6)
        ])
        
        # Output projection
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.use_pos_embedding:
            torch.nn.init.normal_(self.pos_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.output_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        x_flat = x.view(B, h * w)
        seq_len = h * w
        
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        token_emb = self.token_embedding(x_flat)
        
        if self.use_pos_embedding:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
            embeddings = token_emb + pos_emb
        else:
            embeddings = token_emb
        
        shifted_embeddings = torch.zeros_like(embeddings)
        shifted_embeddings[:, 1:] = embeddings[:, :-1]
        
        hidden = shifted_embeddings
        for layer in self.transformer_layers:
            hidden = layer(hidden, src_mask=causal_mask)
        
        logits = self.output_proj(hidden)
        logits = logits.view(B, h, w, self.n_tokens)
        
        return logits, {}
    
    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        generated = torch.zeros(B, h, w, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    logits, _ = self.forward(generated)
                    probs = torch.softmax(logits[:, i, j], dim=-1)
                    next_token = torch.multinomial(probs, 1).squeeze(-1)
                    generated[:, i, j] = next_token
        
        return generated
