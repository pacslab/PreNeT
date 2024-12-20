import torch
import time
import math

class PositionalEncodingBenchmark:
    def __init__(self, 
                 batch_size, 
                 seq_length, 
                 embed_dim, 
                 warmup_iterations, 
                 benchmark_iterations, 
                 device='cuda'):
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.device = device

        self.inputs = torch.zeros(
            self.batch_size, self.seq_length, self.embed_dim, 
            device=self.device, dtype=torch.float32
        )

        # Positional encoding matrix
        self.positional_encoding = self.create_positional_encoding().to(self.device, torch.float32)


    def create_positional_encoding(self):
        pe = torch.zeros(self.seq_length, self.embed_dim)
        position = torch.arange(0, self.seq_length, dtype=torch.float32).unsqueeze(1)

        # Separate div_terms for even and odd indices
        div_term_even = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / self.embed_dim)
        )
        div_term_odd = torch.exp(
            torch.arange(1, self.embed_dim, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / self.embed_dim)
        )

        # Compute positional encodings for even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)

        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)
        return pe


    def _forward_pass(self):
        # Apply positional encoding to inputs
        
        return self.inputs + self.positional_encoding


    def run_benchmark(self):
        # Warm-up iterations
        for _ in range(self.warmup_iterations):
            self._forward_pass()

        if self.device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        # Benchmark iterations
        for _ in range(self.benchmark_iterations):
            self._forward_pass()

        if self.device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # Return average time in ms
        return (end_time - start_time) / self.benchmark_iterations * 1000
