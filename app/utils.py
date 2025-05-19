import os
import pickle
import torch
import numpy as np
import pandas as pd

# Try to register SANFIS as a safe global for PyTorch 2.6+
try:
    from sanfis import SANFIS
    import torch.serialization
    torch.serialization.add_safe_globals([SANFIS])
except (ImportError, AttributeError):
    # Older PyTorch versions don't have this
    pass

def load_models():
    """Load all trained models directly from the Model directory"""
    models = {}
    
    # Load scikit-learn models (KNN, SVM, RF)
    for model_name in ["knn", "svm", "rf"]:
        model_path = os.path.join("..", "Model", f"{model_name}_model.pkl")
        try:
            with open(model_path, "rb") as f:
                models[model_name] = pickle.load(f)
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            models[model_name] = None
    
    # Load ANFIS model (PyTorch) - now expecting a full model, not just state_dict
    anfis_path = os.path.join("..", "Model", "anfis_model.pt")
    try:
        # For PyTorch 2.6+, use weights_only=False
        try:
            models["anfis"] = torch.load(anfis_path, map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            models["anfis"] = torch.load(anfis_path, map_location=torch.device('cpu'))
        print("Successfully loaded ANFIS model")
    except Exception as e:
        print(f"Error loading ANFIS model: {e}")
        # If loading the full model fails, try loading as state_dict with fallback
        try:
            from sanfis import SANFIS
            
            # Try with different weights_only values
            try:
                state_dict = torch.load(anfis_path, map_location=torch.device('cpu'), weights_only=False)
            except TypeError:
                state_dict = torch.load(anfis_path, map_location=torch.device('cpu'))
            
            # Try to load config file if it exists
            config_path = os.path.join("..", "Model", "anfis_config.pkl")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
                    anfis_model = SANFIS(**config)
            else:
                # Default configuration if config file not found
                anfis_model = SANFIS(
                    membfuncs=[
                        {'function': 'gaussian', 'n_memb': 7, 
                         'params': {'mu': {'value': np.linspace(-1, 1, 7).tolist(), 'trainable': True},
                                   'sigma': {'value': [0.8]*7, 'trainable': True}}},
                        {'function': 'bell', 'n_memb': 7,
                         'params': {'c': {'value': np.linspace(-1, 1, 7).tolist(), 'trainable': True},
                                   'a': {'value': [2.0]*7, 'trainable': False},
                                   'b': {'value': [3.0]*7, 'trainable': False}}},
                        {'function': 'gaussian', 'n_memb': 7,
                         'params': {'mu': {'value': np.linspace(-1, 1, 7).tolist(), 'trainable': True},
                                   'sigma': {'value': [0.8]*7, 'trainable': True}}},
                        {'function': 'sigmoid', 'n_memb': 7,
                         'params': {'c': {'value': np.linspace(-1, 1, 7).tolist(), 'trainable': True},
                                   'gamma': {'value': [-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0], 'trainable': True}}}
                    ],
                    n_input=4,
                    scale='Std'
                )
            
            # If state_dict is already a dict, use it directly
            if isinstance(state_dict, dict):
                anfis_model.load_state_dict(state_dict)
            # If it's the old model, extract state_dict
            elif hasattr(state_dict, 'state_dict'):
                anfis_model.load_state_dict(state_dict.state_dict())
            
            models["anfis"] = anfis_model
            print("Successfully loaded ANFIS model from state_dict")
        except Exception as nested_e:
            print(f"Failed fallback for ANFIS model: {nested_e}")
            models["anfis"] = None
    
    return models

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height (cm) and weight (kg)"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 1)

def estimate_dpf(parent_diabetes, sibling_diabetes, other_relatives):
    """Estimate Diabetes Pedigree Function based on family history
    
    This is a simplified estimation and not a medical calculation.
    """
    base_score = 0.172  # Median value from the dataset
    
    if parent_diabetes:
        base_score += 0.372
    if sibling_diabetes:
        base_score += 0.252
    if other_relatives:
        base_score += 0.148
    
    return min(base_score, 2.4)  # Cap at reasonable maximum

def get_risk_category(probability):
    """Convert probability to risk category"""
    if probability < 0.3:
        return "Low Risk", "low"
    elif probability < 0.7:
        return "Medium Risk", "medium" 
    else:
        return "High Risk", "high"

def get_prediction(model, features):
    """Get prediction from the specified model"""
    # If model failed to load, return a middle-range prediction
    if model is None:
        return 0.5
    
    features_array = np.array(features).reshape(1, -1)
    
    try:
        if isinstance(model, torch.nn.Module):
            # ANFIS model
            tensor_input = torch.tensor(features_array, dtype=torch.float32)
            with torch.no_grad():
                # Check available parameters in the forward method
                if hasattr(model, 'forward'):
                    param_names = getattr(model.forward, '__code__', None)
                    if param_names and hasattr(param_names, 'co_varnames'):
                        param_names = param_names.co_varnames
                        
                        # Try different parameter names used by different SANFIS versions
                        if 'S_batch' in param_names:
                            output = model(S_batch=tensor_input)
                        elif 'X_batch' in param_names:
                            output = model(X_batch=tensor_input)
                        else:
                            # Default positional parameter
                            output = model(tensor_input)
                    else:
                        # If we can't inspect parameters, try positional
                        output = model(tensor_input)
                else:
                    # No forward method, try calling directly
                    output = model(tensor_input)
                
            probability = output.item() if isinstance(output, torch.Tensor) else output[0][0]
        else:
            # Scikit-learn models
            probability = model.predict_proba(features_array)[0][1]
    except Exception as e:
        print(f"Error making prediction: {e}")
        # Return a simple heuristic based on features
        probability = calculate_simple_risk(features)
    
    return probability

def calculate_simple_risk(features):
    """Calculate a simple risk score based on the features"""
    pregnancies, glucose, bmi, dpf = features
    
    # Normalize and combine features
    preg_score = min(pregnancies / 8.0, 1.0)
    glucose_score = (glucose - 70) / (180 - 70) if glucose is not None else 0.5
    bmi_score = (bmi - 18.5) / (40 - 18.5) if bmi > 18.5 else 0
    dpf_score = min(dpf / 1.5, 1.0)
    
    # Weight each factor (glucose is most important)
    weighted_score = (0.15 * preg_score + 
                      0.45 * glucose_score + 
                      0.25 * bmi_score + 
                      0.15 * dpf_score)
    
    return weighted_score

def get_factor_contributions(features, feature_names):
    """Get feature importance for explaining results"""
    # Normalized feature values (higher is more concerning)
    # These thresholds are based on medical guidelines and dataset distribution
    try:
        norm_features = {
            "Pregnancies": min(features[0] / 8, 1) if features[0] is not None else 0.5,
            "Glucose": (features[1] - 70) / (180 - 70) if features[1] is not None else 0.5,  # 70-180 range
            "BMI": (features[2] - 18.5) / (40 - 18.5) if features[2] > 18.5 else 0,  # 18.5-40 range
            "DiabetesPedigreeFunction": min(features[3] / 1.5, 1) if features[3] is not None else 0.5  # 0-1.5 range
        }
    except (IndexError, TypeError, ValueError) as e:
        # If there's any issue with the calculation, return default values
        print(f"Warning: Error in factor contributions calculation: {e}")
        norm_features = {
            "Pregnancies": 0.5,
            "Glucose": 0.5,
            "BMI": 0.5,
            "DiabetesPedigreeFunction": 0.5
        }
    
    return norm_features 