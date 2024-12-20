import os
import gc
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd


from benchmark_dense import BenchmarkDense
from benchmark_conv import BenchmarkConv
from benchmark_rnn import BenchmarkRNN
from benchmark_attention import MultiHeadAttentionBenchmark
from benchmark_layernorm import LayerNormBenchmark
from benchmark_embedding import EmbeddingBenchmark
from benchmark_positional_encoding import PositionalEncodingBenchmark


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('Benchmarking different layers')

# Benchmarks to perform
parser.add_argument('--testDense', action="store_true", default=False,
                    help='Benchmark fully connected layer/matrix multiplication')
parser.add_argument('--testConv', action="store_true", default=False,
                    help='Benchmark 2D convolution')
parser.add_argument('--testRNN', action="store_true", default=False,
                    help='Benchmark RNN')
parser.add_argument('--testAttention', action="store_true", default=False,
                    help='Benchmark attention mechanism')
parser.add_argument('--testLayerNorm', action="store_true", default=False,
                    help='Benchmark layer normalization')
parser.add_argument('--testEmbedding', action="store_true", default=False,
                    help='Benchmark embedding layer')
parser.add_argument('--testPositionalEncoding', action="store_true", default=False,
                    help='Benchmark positional endcoding')
# General parameters
parser.add_argument('--backprop_ratio', type=float, default=0.5,
                    help='Ratio of iterations with backward pass ([0..1])')
parser.add_argument('--num_val', type=int, default=100,
                    help='Number of results to compute')
parser.add_argument('--logdir', type=str, default='./',
                    help='Directory to store results')
parser.add_argument('--logfile', type=str, default='',
                    help='Text file to store results')
parser.add_argument('--device', type=str, default='',
                    help='Device name as appearing in logfile')
parser.add_argument('--iter_benchmark', type=int, default=50,
                    help='Number of iterations for benchmark')
parser.add_argument('--iter_warmup', type=int, default=10,
                    help='Number of iterations for warm-up')
parser.add_argument('--repetitions', type=int, default=5,
                    help='Number of repetitions of the same experiment')


args = parser.parse_args()


def main():
    optimizer_list = [
        'None',
        'SGD',
        'Adadelta',
        'Adagrad',
        'Adam',
        'RMSprop'
    ]
    
    activation_list = [
        'None',
        'ReLU',
        'Tanh',
        'Sigmoid'
    ]
    
    
    ########### Benchmark LayerNorm ##########
    if args.testLayerNorm:
        if args.logfile == '':
            logfile = str('%s/benchmark_layernorm_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
            
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
            
        print('Benchmarking LayerNorm...')
        batchsize = np.random.randint(1, 65, size=args.num_val)
        seq_len = np.random.randint(1, 513, size=args.num_val)
        embed_dim = np.random.randint(32, 1025, size=args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        precision = (np.ones(args.num_val) * 32).astype(int)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
        
        for rep in range(args.repetitions):
            print(f"Benchmarking layernorm, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = LayerNormBenchmark(
                        batchsize[i],
                        seq_len[i],
                        embed_dim[i],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished layernorm {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'precision': precision,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')
    
    
    ########### Benchmark Attention ##########
    if args.testAttention:
        if args.logfile == '':
            logfile = str('%s/benchmark_attention_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
            
        print('Benchmarking Attention...')
        batchsize = np.random.randint(1, 65, size=args.num_val)
        seq_len = np.random.randint(1, 257, size=args.num_val)
        num_heads_list = np.random.randint(1, 13, size=args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        precision = (np.ones(args.num_val) * 32).astype(int)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
                
        embed_dim = []
        for i in range(args.num_val):
            min_embed_dim = num_heads_list[i] * 8  # Minimum to ensure divisibility
            max_embed_dim = 768  # Max embed_dim = 768, common in BERT base
            em_dim = random.randint(min_embed_dim, max_embed_dim)
            em_dim = (em_dim // num_heads_list[i]) * num_heads_list[i]
            embed_dim.append(em_dim)
        embed_dim = np.array(embed_dim)
            
        
        for rep in range(args.repetitions):
            print(f"Benchmarking attention, starting repetition {rep}")
            for i in range(args.num_val):
                
                try:
                    timeUsed[i, rep] = MultiHeadAttentionBenchmark(
                        batchsize[i],
                        seq_len[i],
                        embed_dim[i],
                        num_heads_list[i],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished attention {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'num_heads': num_heads_list,
            'precision': precision,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')

    
    ########### Benchmark Positional Encoding ##########
    if args.testPositionalEncoding:
        if args.logfile == '':
            logfile = str('%s/benchmark_positional_encoding_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
            
        print('Benchmarking Positional Encoding...')
        batchsize = np.random.randint(1, 65, size=args.num_val)
        seq_len = np.random.randint(1, 513, size=args.num_val)
        embed_dim = np.random.randint(32, 1025, size=args.num_val)
        precision = (np.ones(args.num_val) * 32).astype(int)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
                
        for rep in range(args.repetitions):
            print(f"Benchmarking positional encoding, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = PositionalEncodingBenchmark(
                        batchsize[i],
                        seq_len[i],
                        embed_dim[i],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished positional encoding {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
            'precision': precision,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')
    
    
    ########### Benchmark Embedding ##########
    if args.testEmbedding:
        if args.logfile == '':
            logfile = str('%s/benchmark_embedding_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
        
        # Set random parameters
        batchsize = np.random.randint(1, 65, size=args.num_val)
        seq_len = np.random.randint(1, 513, size=args.num_val)
        vocab_size = np.random.randint(1000, 100001, size=args.num_val)
        embed_dim = np.random.randint(32, 1025, size=args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        precision = (np.ones(args.num_val) * 32).astype(int)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
        
        for rep in range(args.repetitions):
            print(f"Benchmarking embedding, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = EmbeddingBenchmark(
                        batchsize[i],
                        seq_len[i],
                        vocab_size[i],
                        embed_dim[i],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished embedding {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'seq_len': seq_len,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'precision': precision,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)
        })
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')
    
    
    ########### Benchmark RNN ##########
    if args.testRNN:
        
        if args.logfile == '':
            logfile = str('%s/benchmark_rnn_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
            
        batchsize = np.random.randint(1, 65, args.num_val)
        seq_len = np.random.randint(1, 129, args.num_val)
        input_dim = np.random.randint(1, 129, args.num_val, dtype=np.int32)
        hidden_dim = np.random.randint(1, 129, args.num_val, dtype=np.int32)
        num_layers = np.random.randint(1, 5, args.num_val)
        rnn_type = np.random.choice(['LSTM', 'GRU', 'RNN'], args.num_val)
        is_bidirectional = np.random.choice([True, False], args.num_val)
        activation_fct = np.random.randint(0, len(activation_list), args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        precision = (np.ones(args.num_val) * 32).astype(int)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
                
        for rep in range(args.repetitions):
            print(f"Benchmarking RNN, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = BenchmarkRNN(
                        batchsize[i],
                        seq_len[i],
                        input_dim[i],
                        hidden_dim[i],
                        is_bidirectional[i],
                        num_layers[i],
                        rnn_type[i],
                        activation_list[activation_fct[i]],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished RNN {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'seq_len': seq_len,
            'dim_input': input_dim,
            'dim_hidden': hidden_dim,
            'precision': precision,
            'is_bidirectional': is_bidirectional,
            'num_layers': num_layers,
            'rnn_type': rnn_type,
            'activation_fct': activation_fct,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')
    
    
    ########### Benchmark convolution ##########
    if args.testConv:
        if args.logfile == '':
            logfile = str('%s/benchmark_convolution_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)

        # Set random parameters
        batchsize = np.random.randint(1, 65, args.num_val)
        matsize = np.random.randint(1, 513, args.num_val)
        kernelsize = np.zeros(args.num_val, dtype=np.int32)
        channels_in = np.zeros(args.num_val, dtype=np.int32)
        channels_out = np.zeros(args.num_val, dtype=np.int32)
        strides = np.random.randint(1, 5, args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        precision = (np.ones(args.num_val) * 32).astype(int)
        padding = np.random.choice(['valid', 'same'], args.num_val)
        activation_fct = np.random.randint(0, len(activation_list), args.num_val)
        use_bias = np.random.choice([True, False], args.num_val)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            kernelsize[i] = np.random.randint(1, min(7, matsize[i]) + 1)
            channels_in[i] = np.random.randint(1, 10000 // matsize[i])
            channels_out[i] = np.random.randint(1, 10000 // matsize[i])
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
        
        for rep in range(args.repetitions):
            print(f"Benchmarking convolution, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = BenchmarkConv(
                        batchsize[i],
                        matsize[i],
                        kernelsize[i],
                        channels_in[i],
                        channels_out[i],
                        strides[i],
                        ('same' if padding[i] == 1 else 'valid'),
                        activation_list[activation_fct[i]],
                        use_bias[i],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished convolution {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'matsize': matsize,
            'kernelsize': kernelsize,
            'channels_in': channels_in,
            'channels_out': channels_out,
            'strides': strides,
            'padding': padding,
            'precision': precision,
            'activation_fct': activation_fct,
            'use_bias': use_bias,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')
    
    ########### Benchmark fully connected layer ##########
    if args.testDense:
        if args.logfile == '':
            logfile = str('%s/benchmark_dense_%s_%s'
                          % (args.logdir, args.device, time.strftime("%Y%m%d")))
        else:
            logfile = '%s/%s' % (args.logdir, args.logfile)
        
        # Set random parameters
        batchsize = np.random.randint(1, 65, args.num_val)
        dim_input = np.random.randint(1, 4096, args.num_val)
        dim_output = np.random.randint(1, 4096, args.num_val)
        precision = (np.ones(args.num_val) * 32).astype(int)
        activation_fct = np.random.randint(0, len(activation_list), args.num_val)
        optimizer = np.zeros(args.num_val, dtype=np.int32)
        
        timeUsed = np.zeros([args.num_val, args.repetitions])
        
        for i in range(args.num_val):
            if np.random.rand() <= args.backprop_ratio:
                optimizer[i] = np.random.randint(1, len(optimizer_list))
            else:
                optimizer[i] = 0
        
        for rep in range(args.repetitions):
            print(f"Benchmarking fully connected, starting repetition {rep}")
            for i in range(args.num_val):
                try:
                    timeUsed[i, rep] = BenchmarkDense(
                        batchsize[i],
                        dim_input[i],
                        dim_output[i],
                        activation_list[activation_fct[i]],
                        optimizer_list[optimizer[i]],
                        args.iter_warmup,
                        args.iter_benchmark,
                        device,
                    ).run_benchmark()

                except RuntimeError as e:
                    print(f'Error: {e}')
                    timeUsed[i, rep] = None
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}/{args.num_val}: Finished dense layer {i + 1}/{args.num_val} (Rep {rep + 1}/{args.repetitions})")
                
                gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save results
        print("Saving results")
        df_results = pd.DataFrame({
            'batchsize': batchsize,
            'dim_input': dim_input,
            'dim_output': dim_output,
            'precision': precision,
            'activation_fct': activation_fct,
            'optimizer': optimizer,
            'timeUsed_median': np.median(timeUsed, 1),
            'timeUsed_min': np.min(timeUsed, 1),
            'timeUsed_max': np.max(timeUsed, 1),
            'timeUsed_std': np.std(timeUsed, 1)})
        
        print(df_results)
        df_results.to_csv(f'{logfile}.csv')


if __name__ == '__main__':
    main()
