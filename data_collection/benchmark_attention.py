import torch
import time

class MultiHeadAttentionBenchmark:
    def __init__(self, 
                 batch_size, 
                 seq_length, 
                 embed_dim, 
                 num_heads, 
                 optimizer,
                 warmup_iterations, 
                 benchmark_iterations, 
                 device='cuda'):
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.device = device

        self.optimizer_fn = None
        self.criterion = None

        # Initialize the Multi-Head Attention model
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            batch_first=True
        ).to(self.device, torch.float32)

        self.attention.train()

        # Set optimizer and criterion if optimizer is provided
        if self.optimizer != 'None':
            self.optimizer_fn = getattr(torch.optim, optimizer)(self.attention.parameters(), lr=0.0001)
            self.criterion = torch.nn.MSELoss()


    def _forward_pass(self):
        x = torch.randn(
            self.batch_size, self.seq_length, self.embed_dim,
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        y, _ = self.attention(x, x, x)
        
        return y


    def _backward_pass(self, y):
        target = torch.randn_like(y)
        loss = self.criterion(y, target)
        self.optimizer_fn.zero_grad()
        loss.backward()
        self.optimizer_fn.step()
        
        return loss


    def run_benchmark(self):
        # Warm-up iterations
        for _ in range(self.warmup_iterations):
            y = self._forward_pass()
            if self.optimizer_fn:
                self._backward_pass(y)

        # Synchronize and measure time for actual benchmarking
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(self.benchmark_iterations):
            y = self._forward_pass()
            if self.optimizer_fn:
                self._backward_pass(y)

        if self.device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # Return time in ms
        return (end_time - start_time) / self.benchmark_iterations * 1000
