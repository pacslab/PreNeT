import torch
import time

class BenchmarkRNN:
    def __init__(self,
                 batchsize,
                 seq_len,
                 input_dim,
                 hidden_dim,
                 is_bidirectional,
                 num_layers,
                 rnn_type,
                 activation,
                 optimizer,
                 warmup_iterations,
                 benchmark_iterations,
                 device='cuda'):
        
        self.batchsize = batchsize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_bidirectional = is_bidirectional
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.activation = activation
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.device = device
        
        # Initialize the RNN (LSTM, GRU, or RNN)
        if self.rnn_type in ['LSTM', 'GRU', 'RNN']:
            rnn_class = getattr(torch.nn, self.rnn_type)
            self.rnn = rnn_class(self.input_dim, int(self.hidden_dim), 
                                 num_layers=self.num_layers, 
                                 bidirectional=bool(self.is_bidirectional), 
                                 batch_first=True).to(self.device)
            
        self.activation_fn = None
        self.optimizer_fn = None
        self.criterion = None

        # Initialize the activation function, if specified
        if self.activation != 'None':
            self.activation_fn = getattr(torch.nn, self.activation)().to(self.device)


        # Initialize the optimizer and loss function, if specified
        if self.optimizer != 'None':
            self.criterion = torch.nn.MSELoss()
            self.optimizer_fn = getattr(torch.optim, self.optimizer)(self.rnn.parameters(), lr=0.0001)


    def _forward_pass(self):
        x = torch.randn(self.batchsize, self.seq_len, self.input_dim, device=self.device)
        rnn_out, _ = self.rnn(x)
        y = self.activation_fn(rnn_out[:, -1, :]) if self.activation_fn else rnn_out[:, -1, :]
        
        return y


    def _backward_pass(self, y):
        target = torch.randn(self.batchsize, self.hidden_dim * (2 if self.is_bidirectional else 1), device=self.device)
        self.optimizer_fn.zero_grad()
        loss = self.criterion(y, target)
        loss.backward()
        self.optimizer_fn.step()
        
        return loss


    def run_benchmark(self):
        # Warm-up phase
        for _ in range(self.warmup_iterations):
            y = self._forward_pass()
            if self.optimizer_fn:
                self._backward_pass(y)

        # Benchmark phase
        if self.device == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU timing accuracy

        start_time = time.time()
        for _ in range(self.benchmark_iterations):
            y = self._forward_pass()
            if self.optimizer_fn:
                self._backward_pass(y)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # Return time in ms per iteration
        return (end_time - start_time) / self.benchmark_iterations * 1000