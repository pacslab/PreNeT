from enum import Enum
from typing import Union, Dict, Any

import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


ACTIVATIONS = ['None', 'relu', 'tanh', 'sigmoid']
OPTIMIZERS = ['None', 'SGD', 'Adadelta', 'Adagrad', 'Adam', 'RMSProp']
RNN_TYPES = ['LSTM', 'GRU', 'RNN']
PADDINGS = ['valid', 'same']
BUS = ['PCIe 3.0', 'PCIe 4.0']
MEMORY_TYPES = ['GDDR5', 'GDDR6', 'GDDR6X', 'HBM2']


class GPUFeatures(Enum):
    BASE_CLOCK_GHZ = 'Base Clock (MHz)'
    BOOST_CLOCK_GHZ = 'Boost Clock (MHz)'
    MEMORY_TYPE = 'Memory Type'
    MEMORY_BUS_BIT = 'Memory Bus (bit)'
    MEMORY_CLOCK_GHZ = 'Memory Clock (MHz)'
    MEMORY_SIZE_GB = 'Memory (GB)'
    MEMORY_BANDWIDTH_GBS = 'GPU Memory Bandwidth (GB/s)'
    BUS = 'Bus'
    CORES = 'Cores'
    TC = 'TC'
    RT = 'RT'
    FP32 = 'FP32'
    GPU_NAME = 'GPU'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]
    
    
class ConvFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    MAT_SIZE = 'matsize'
    KERNEL_SIZE = 'kernelsize'
    CHANNELS_IN = 'channels_in'
    CHANNELS_OUT = 'channels_out'
    STRIDES = 'strides'
    PADDING = 'padding'
    PRECISION = 'precision'
    ACTIVATION = 'activation_fct'
    USE_BIAS = 'use_bias'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]


class DenseFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    DIM_INPUT = 'dim_input'
    DIM_OUTPUT = 'dim_output'
    PRECISION = 'precision'
    ACTIVATION = 'activation_fct'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]
    
    
class RNNFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    SEQ_LEN = 'seq_len'
    DIM_INPUT = 'dim_input'
    DIM_HIDDEN = 'dim_hidden'
    PRECISION = 'precision'
    NUM_LAYERS = 'num_layers'
    IS_BIDIRECTIONAL = 'is_bidirectional'
    RNN_TYPE = 'rnn_type'
    ACTIVATION = 'activation_fct'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]
    
    
class AttentionFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    SEQ_LEN = 'seq_len'
    EMBED_DIM = 'embed_dim'
    NUM_HEADS = 'num_heads'
    PRECISION = 'precision'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]


class EmbeddingFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    SEQ_LEN = 'seq_len'
    VOCAB_SIZE = 'vocab_size'
    EMBEDDING_DIM = 'embed_dim'
    PRECISION = 'precision'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]
    
    
class LayerNormFeatures(Enum):
    BATCH_SIZE = 'batchsize'
    SEQ_LEN = 'seq_len'
    EMBEDDING_DIM = 'embed_dim'
    PRECISION = 'precision'
    OPTIMIZER = 'optimizer'
    
    @classmethod
    def get_features(cls):
        return [column.value for column in cls]


class Preprocessor:
    def __init__(self):
        self.features_list = None
        self.features = None
        self._is_preprocessed = False
    
    @staticmethod   
    def _preprocess(self):
        pass
    
    def get_features_as_tensors(self):
        if not self.features:
            return

        self._preprocess()
        
        return torch.tensor(self.features.values, dtype=torch.float32)
    
    def get_features_list(self):
        if not self.features_list:
            return

        return self.features.columns.tolist()
    
    
class PreprocessAttentionFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = AttentionFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()
        
    def _preprocess(self):
        if self._is_preprocessed:
            return
        
        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})
        
        
        comp_ops_projection = self.features['batchsize'] * (self.features['seq_len'] * self.features['embed_dim'] ** 2)
        comp_ops_qk = self.features['batchsize'] * (self.features['seq_len'] ** 2 * self.features['embed_dim'])
        comp_ops_softmax = self.features['batchsize'] * (self.features['seq_len'] ** 2)
        comp_ops_weighted_sum = self.features['batchsize'] * (self.features['seq_len'] ** 2 * self.features['embed_dim'])
        comp_ops_output_projection = self.features['batchsize'] * (self.features['seq_len'] * self.features['embed_dim'] ** 2)
        
        self.features['flops'] = self.features['num_heads'] * (comp_ops_projection + 2 * comp_ops_qk + comp_ops_softmax + comp_ops_weighted_sum + comp_ops_output_projection)
        
        if self._include_additional_features:
            self.features['memory_projection'] = 3 * self.features['batchsize'] * self.features['seq_len'] * self.features['embed_dim']
            self.features['memory_attention'] = self.features['batchsize'] * self.features['seq_len'] * self.features['seq_len']
            self.features['memory_out'] = self.features['batchsize'] * self.features['seq_len'] * self.features['embed_dim']
            
            self.features['memory_total'] = self.features['memory_projection'] + self.features['memory_attention'] + self.features['memory_out']            
            self.features['memory_total'] = np.where(self.features['optimizer'] != 'None', self.features['memory_total'] + 2 * (self.features['memory_projection'] + self.features['memory_attention'] + self.features['memory_out']), self.features['memory_total'])

            self.features['flops/speed'] = self.features['flops'] / self.features['FP32']
        
        categorical_features = ['Memory Type', 'Bus', 'optimizer']

        encoder = OneHotEncoder(categories=[MEMORY_TYPES, BUS, OPTIMIZERS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])

        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True
    
    
class PreprocessLayerNormFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = LayerNormFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()
        
    def _preprocess(self):
        if self._is_preprocessed:
            return
        
        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})
        
        categorical_features = ['optimizer', 'Memory Type', 'Bus']

        encoder = OneHotEncoder(categories=[OPTIMIZERS, MEMORY_TYPES, BUS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])

        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True


class PreprocessEmbeddingFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = EmbeddingFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()
        
    def _preprocess(self):
        if self._is_preprocessed:
            return
        
        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})
        
        if self._include_additional_features:
            self.features['flops'] = (self.features['batchsize'] * self.features['seq_len'] * self.features['embed_dim'])
        
            self.features['memory_embeddings'] = self.features['vocab_size'] * self.features['embed_dim']
            self.features['memory_in'] = self.features['batchsize'] * self.features['seq_len'] * self.features['embed_dim']
            self.features['memory_total'] = self.features['memory_embeddings'] + self.features['memory_in']
            self.features['memory_total'] = np.where(self.features['optimizer'] != 'None', 2 * self.features['memory_total'], self.features['memory_total'])
            
            self.features['flops/speed'] = self.features['flops'] / self.features['FP32']
        
        categorical_features = ['optimizer', 'Memory Type', 'Bus']

        encoder = OneHotEncoder(categories=[OPTIMIZERS, MEMORY_TYPES, BUS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])

        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True
        
    

class PreprocessDenseFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = DenseFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()


    def _preprocess(self):
        if self._is_preprocessed:
            return

        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})

        self.features.loc[:, 'activation_fct'] = self.features['activation_fct'].map({0:'None',
                                                                                   1:'relu',
                                                                                   2:'tanh',
                                                                                   3:'sigmoid'})
        
        
        if self._include_additional_features:
            self.features['flops'] = (
                self.features['batchsize'] * self.features['dim_input'] * self.features['dim_output'] * 2
            )
            
            self.features['flops/speed'] = self.features['flops'] / self.features['FP32']
            
            self.features['memory_weights'] = self.features['dim_input'] * self.features['dim_output']
            self.features['memory_in'] = self.features['batchsize'] * self.features['dim_input']
            self.features['memory_out'] = self.features['batchsize'] * self.features['dim_output']
            
            self.features['memory_total'] = self.features['memory_in'] + self.features['memory_out'] + self.features['memory_weights']
            self.features['memory_total'] = np.where(self.features['optimizer'] != 'None', 2 * self.features['memory_total'], self.features['memory_total'])
        
        categorical_features = ['activation_fct', 'optimizer', 'Memory Type', 'Bus']

        encoder = OneHotEncoder(categories=[ACTIVATIONS, OPTIMIZERS, MEMORY_TYPES, BUS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])
        
        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features), index=self.features.index)
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True
    
    
class PreprocessConvFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = ConvFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()


    def _preprocess(self):
        if self._is_preprocessed:
            return

        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})

        self.features.loc[:, 'activation_fct'] = self.features['activation_fct'].map({0:'None',
                                                                                   1:'relu',
                                                                                   2:'tanh',
                                                                                   3:'sigmoid'})
        
        if self._include_additional_features:
            padding_reduction = ((self.features['padding'] == 'valid')*(self.features['kernelsize']-1))

            elements_output = ((self.features['matsize'] - padding_reduction) / self.features['strides'])**2

            self.features['flops'] = (self.features['batchsize']
                * elements_output
                * self.features['kernelsize']**2
                * self.features['channels_in']
                * self.features['channels_out'])
            
            self.features['flops/speed'] = self.features['flops'] / self.features['FP32']
            
            self.features['memory_weights'] = (self.features['kernelsize']**2
                    * self.features['channels_in']
                    * self.features['channels_out']
                    + self.features['use_bias'] * self.features['channels_out'])

            self.features['memory_in'] = (self.features['batchsize']
                        * self.features['matsize']**2
                        * self.features['channels_in'])

            self.features['memory_out'] = (self.features['batchsize']
                        * elements_output
                        * self.features['channels_out'])
            
            self.features['memory_total'] = self.features['memory_in'] + self.features['memory_out'] + self.features['memory_weights']
            self.features['memory_total'] = np.where(self.features['optimizer'] != 'None', 2 * self.features['memory_total'], self.features['memory_total'])
        
        self.features['use_bias'] = self.features['use_bias'].astype(int)
        
        categorical_features = ['activation_fct', 'optimizer', 'padding', 'Memory Type', 'Bus']

        encoder = OneHotEncoder(categories=[ACTIVATIONS, OPTIMIZERS, PADDINGS, MEMORY_TYPES, BUS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])

        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True
    
    
    
class PreprocessRNNFeatures(Preprocessor):
    def __init__(self, features: Union[Dict[str, Any], pd.DataFrame], include_additional_features: bool = False):
        super().__init__()
        self.features_list = RNNFeatures.get_features() + GPUFeatures.get_features()
        self._is_preprocessed = False
        self._include_additional_features = include_additional_features
        
        try:
            if isinstance(features, pd.DataFrame):
                self.features = features[self.features_list]
            else:
                self.features = pd.DataFrame(features, index=[0])[self.features_list]

        except KeyError as e:
            raise KeyError(f'KeyError: {e}. Please provide all the required features.')
        
        self._preprocess()


    def _preprocess(self):
        if self._is_preprocessed:
            return

        self.features['is_bidirectional'] = np.uint8(self.features['is_bidirectional'])
        
        self.features.loc[:, 'optimizer'] = self.features['optimizer'].map({0:'None',
                                                                         1:'SGD',
                                                                         2:'Adadelta',
                                                                         3:'Adagrad',
                                                                         4:'Adam',
                                                                         5:'RMSProp'})

        self.features.loc[:, 'activation_fct'] = self.features['activation_fct'].map({0:'None',
                                                                                   1:'relu',
                                                                                   2:'tanh',
                                                                                   3:'sigmoid'})
        
        if self._include_additional_features:
            batchsize = self.features['batchsize'].values
            seq_len = self.features['seq_len'].values
            dim_input = self.features['dim_input'].values
            dim_hidden = self.features['dim_hidden'].values

            lstm_ops = (batchsize * seq_len * (4 * (2 * (dim_input * dim_hidden + dim_hidden ** 2) + dim_hidden) + 2 * dim_hidden))
            gru_ops = (batchsize * seq_len * (3 * (2 * (dim_input * dim_hidden + dim_hidden ** 2) + dim_hidden) + 2 * dim_hidden))
            rnn_ops = (batchsize * seq_len * (2 * (dim_input * dim_hidden + dim_hidden ** 2) + dim_hidden))

            self.features['flops'] = np.where(self.features['rnn_type'] == 'LSTM', lstm_ops,
                                            np.where(self.features['rnn_type'] == 'GRU', gru_ops, rnn_ops))

            self.features['flops'] = np.where(self.features['is_bidirectional'] == 1, 2 * self.features['flops'], self.features['flops'])

            self.features['flops'] = np.where(self.features['optimizer'] != 'None', 3 * self.features['flops'], self.features['flops'])

            # Memory Calculation
            lstm_memory_weights = 4 * (dim_input * dim_hidden + dim_hidden ** 2)
            gru_memory_weights = 3 * (dim_input * dim_hidden + dim_hidden ** 2)
            rnn_memory_weights = (dim_input * dim_hidden + dim_hidden ** 2)
            
            # Calculate memory_in, memory_out, and memory_weights for each type
            self.features['memory_in'] = batchsize * seq_len * dim_input
            self.features['memory_out'] = batchsize * seq_len * dim_hidden
            self.features['memory_weights'] = np.where(self.features['rnn_type'] == 'LSTM', lstm_memory_weights,
                                                    np.where(self.features['rnn_type'] == 'GRU', gru_memory_weights, rnn_memory_weights))

            self.features['memory_weights'] = np.where(self.features['is_bidirectional'] == 1, 2 * self.features['memory_weights'], self.features['memory_weights'])
            self.features['memory_weights'] = np.where(self.features['optimizer'] != 'None', 2 * self.features['memory_weights'], self.features['memory_weights'])
            self.features['memory_total'] = self.features['memory_in'] + self.features['memory_out'] + self.features['memory_weights']

            self.features['flops/speed'] = self.features['flops'] / self.features['FP32']
        
        categorical_features = ['activation_fct', 'optimizer', 'rnn_type', 'Memory Type', 'Bus']

        encoder = OneHotEncoder(categories=[ACTIVATIONS, OPTIMIZERS, RNN_TYPES, MEMORY_TYPES, BUS], drop=None, sparse_output=False)
        encoded_features = encoder.fit_transform(self.features[categorical_features])

        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        self.features = pd.concat([self.features.drop(categorical_features, axis=1), encoded_data], axis=1)
        
        self.features = self.features[sorted(self.features.columns)]
        
        self._is_preprocessed = True