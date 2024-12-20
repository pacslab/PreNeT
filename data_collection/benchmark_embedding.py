import torch
import time


class EmbeddingBenchmark:
    def __init__(self, 
                 batch_size, 
                 seq_length, 
                 vocab_size, 
                 embed_dim,
                 optimizer,
                 warmup_iterations,
                 benchmark_iterations,
                 device='cuda'):
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.device = device
        
        self.optimizer_fn = None
        self.criterion = None

        # Initialize the embedding layer
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.embed_dim
        ).to(self.device, torch.float32)

        self.embedding.train()

        if optimizer != 'None':
            self.optimizer_fn = getattr(torch.optim, optimizer)(self.embedding.parameters(), lr=0.0001)
            self.criterion = torch.nn.MSELoss()
            
            
    def _forward_pass(self):
        x = torch.randint(
            low=0, high=self.vocab_size,
            size=(self.batch_size, self.seq_length),
            device=self.device
        )
        y = self.embedding(x)
        
        return y
    
    
    def _backward_pass(self, y):
        target = torch.randn(self.batch_size, self.seq_length, self.embed_dim,
                             device=self.device, dtype=torch.float32)
        self.optimizer_fn.zero_grad()
        loss = self.criterion(y, target)
        loss.backward()
        self.optimizer_fn.step()
        
        return loss


    def run_benchmark(self):
        # Warm-up iterations
        for _ in range(self.warmup_iterations):
            y = self._forward_pass()
            if self.optimizer_fn:
                self._backward_pass(y)

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