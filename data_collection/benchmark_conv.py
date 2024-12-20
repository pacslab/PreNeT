import torch
import numpy as np
import time

class BenchmarkConv:
    def __init__(self,
                 batchsize,
                 matsize,
                 kernelsize,
                 channels_in,
                 channels_out,
                 strides,
                 padding,
                 activation,
                 use_bias,
                 optimizer,
                 warmup_iterations,
                 benchmark_iterations,
                 device='cuda'):

        self.batch_size = batchsize
        self.matsize = matsize
        self.kernelsize = kernelsize
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.device = device
        
        self.activation_fn = None
        self.optimizer_fn = None
        self.criterion = None

        # Create the convolutional layer
        self.conv = torch.nn.Conv2d(self.channels_in, self.channels_out, 
                                    kernel_size=self.kernelsize, 
                                    stride=self.strides, 
                                    padding=self.padding, 
                                    bias=self.use_bias).to(self.device)

        # Activation function, if specified
        if self.activation != 'None':
            self.activation_fn = getattr(torch.nn, self.activation)().to(self.device)

        # Optimizer and loss function, if specified
        if self.optimizer != 'None':
            self.optimizer_fn = getattr(torch.optim, self.optimizer)(self.conv.parameters(), lr=0.0001)
            self.criterion = torch.nn.MSELoss()

        # Calculate target output size based on padding
        if self.padding == 'same':
            self.target_size = np.ceil((self.matsize / self.strides)).astype(int)
        else:
            self.target_size = np.ceil((self.matsize - self.kernelsize + 1) / self.strides).astype(int)


    def _forward_pass(self):
        x = torch.randn(self.batch_size, self.channels_in, self.matsize, self.matsize, device=self.device)
        y = self.conv(x)
        
        if self.activation_fn:
            y = self.activation_fn(y)

        return y


    def _backward_pass(self, y):
        target = torch.randn(self.batch_size, self.channels_out, self.target_size, self.target_size, device=self.device)
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