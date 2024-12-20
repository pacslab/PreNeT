import joblib
import torch
import pandas as pd
import gdown
import zipfile
import os


class Predictor:
    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(self.script_dir, 'models')
        
        if not os.path.exists(self.models_dir):
            print(f"{self.models_dir} not found. Downloading and extracting models...")

            gdown.download('https://drive.google.com/uc?id=1alcdWRvUXGwXzmd-MRdxF73YIiwn-XFb&export=download',
                           f'{self.models_dir}.zip',
                           quiet=False)

            with zipfile.ZipFile(f'{self.models_dir}.zip', 'r') as zip_ref:
                zip_ref.extractall(self.script_dir)
                
            os.remove(f'{self.models_dir}.zip')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def predict(self, features):
        pass


class DensePredictor(Predictor):
    def __init__(self, with_features=True):
        # The predictor for this layer is a random forest model
        super().__init__()
        model_path = f'{self.models_dir}/dense_random_forest.pkl' if with_features else f'{self.models_dir}/dense_random_forest_wo_features.pkl'
        scaler_path = f'{self.models_dir}/dense_scaler.pkl' if with_features else f'{self.models_dir}/dense_scaler_wo_features.pkl'
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        return self.model.predict(features)
    
    
class ConvPredictor(Predictor):
    def __init__(self, with_features=True):
        # The predictor for this layer is a random forest model
        super().__init__()
        model_path = f'{self.models_dir}/conv_random_forest.pkl' if with_features else f'{self.models_dir}/conv_random_forest_wo_features.pkl'
        scaler_path = f'{self.models_dir}/conv_scaler.pkl' if with_features else f'{self.models_dir}/conv_scaler_wo_features.pkl'
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        return self.model.predict(features)
    
    
class RNNPredictor(Predictor):
    def __init__(self, with_features=True):
        # The predictor for this layer is an MLP model
        super().__init__()
        model_path = f'{self.models_dir}/rnn_mlp.pth' if with_features else f'{self.models_dir}/rnn_mlp_wo_features.pth'
        scaler_path = f'{self.models_dir}/rnn_scaler.pkl' if with_features else f'{self.models_dir}/rnn_scaler_wo_features.pkl'
        
        self.model = torch.load(model_path).to(self.device)
        self.scaler = joblib.load(scaler_path)
        
        
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        features = torch.tensor(features.to_numpy(), dtype=torch.float32).to(self.device)
        return self.model(features)
    
    
class AttentionPredictor(Predictor):
    def __init__(self, with_features=True):
        # The predictor for this layer is an MLP model
        super().__init__()
        model_path = f'{self.models_dir}/attention_mlp.pth' if with_features else f'{self.models_dir}/attention_mlp_wo_features.pth'
        scaler_path = f'{self.models_dir}/attention_scaler.pkl' if with_features else f'{self.models_dir}/attention_scaler_wo_features.pkl'
        
        self.model = torch.load(model_path).to(self.device)
        self.scaler = joblib.load(scaler_path)
        
        
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        features = torch.tensor(features.to_numpy(), dtype=torch.float32).to(self.device)
        return self.model(features)
    
    
class EmbeddingPredictor(Predictor):
    def __init__(self, with_features=True):
        # The predictor for this layer is an MLP model
        super().__init__()
        model_path = f'{self.models_dir}/embedding_mlp.pth' if with_features else f'{self.models_dir}/embedding_mlp_wo_features.pth'
        scaler_path = f'{self.models_dir}/embedding_scaler.pkl' if with_features else f'{self.models_dir}/embedding_scaler_wo_features.pkl'
        
        self.model = torch.load(model_path).to(self.device)
        self.scaler = joblib.load(scaler_path)
  
        
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        features = torch.tensor(features.to_numpy(), dtype=torch.float32).to(self.device)
        return self.model(features)
    
    
class LayerNormPredictor(Predictor):
    def __init__(self):
        # The predictor for this layer is an MLP model
        super().__init__()
        model_path = f'{self.models_dir}/layer_norm_mlp.pth'
        scaler_path = f'{self.models_dir}/layer_norm_scaler.pkl'
        
        self.model = torch.load(model_path).to(self.device)
        self.scaler = joblib.load(scaler_path)
        
    
    def predict(self, features):
        features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
        features = torch.tensor(features.to_numpy(), dtype=torch.float32).to(self.device)
        return self.model(features)