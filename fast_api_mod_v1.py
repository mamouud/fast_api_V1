# Library imports
from fastapi import FastAPI, Body
import joblib
import pandas as pd
import shap

# Create a FastAPI instance
app= FastAPI()

# Import serialized data
model = joblib.load('./resources/pipeline_model_test_flask.joblib')
optimum_threshold = joblib.load('./resources/optimum_threshold_mod_v1.joblib')


# Endpoints
@app.get('/')
def index():
    """
    Welcome message.
    Args:  
    - None.  
    Returns:  
    - Message (string).  
    """
    return 'Hello, you are accessing an API'


@app.get('/optimum_threshold/')
def get_optimum_threshold():
    """
    Returns the optimum threshold for the probability of default 
    (pre-calculated). Optimized for the buisness cost function.
    Args:
    - None.
    Returns:
    - optimum_threshold (float).
    """
    return optimum_threshold


@app.get('/prediction/')
def get_prediction(json_client: dict = Body({})):
    """
    Calculates the probability of default for a client.  
    Args:  
    - client data (json).  
    Returns:  
    - probability of default (dict).
    """
    df_one_client = pd.Series(json_client).to_frame().transpose()
    probability = model.predict_proba(df_one_client)[:, 1][0]
    return {'probability': probability}


@app.get('/shap/')
def get_shap(json_client: dict = Body({})):
    """
    Calculates the probability of default for a client.  
    Args:  
    - client data (json).  
    Returns:  
    - SHAP values (json).
    """
    df_one_client = pd.Series(json_client).to_frame().transpose()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_one_client)
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': df_one_client.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)
    return df_shap.to_json(orient='index')
