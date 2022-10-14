from fastapi import FastAPI, Query
import tensorflow as tf
from pathlib import Path
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from sunspots.utils import model_forecast

if not GlobalHydra().is_initialized():
    config_file_path = Path.cwd() / '../../config'
    hydra.initialize_config_dir(config_dir=str(config_file_path), version_base=None)
params = hydra.compose(config_name='config.yaml')

"""
curl -X 'GET' \
  'http://127.0.0.1:8000/prediction/?sample=1.1%2C0.4%2C0.5%2C1.5%2C6.2%2C0.2%2C1.5%2C5.2%2C0.2%2C5.8%2C6.1%2C7.5%2C0.6%2C14.6%2C34.5%2C23.1%2C10.4%2C8.2%2C17.2%2C24.5%2C21.2%2C25.%2C34.3%2C22.%2C51.3%2C37.4%2C34.8%2C67.5%2C55.3%2C60.9' \
  -H 'accept: application/json'
"""

"""
http://127.0.0.1:8000/prediction/?sample=1.1%2C0.4%2C0.5%2C1.5%2C6.2%2C0.2%2C1.5%2C5.2%2C0.2%2C5.8%2C6.1%2C7.5%2C0.6%2C14.6%2C34.5%2C23.1%2C10.4%2C8.2%2C17.2%2C24.5%2C21.2%2C25.%2C34.3%2C22.%2C51.3%2C37.4%2C34.8%2C67.5%2C55.3%2C60.9
"""

app = FastAPI()
model_file = 'data/trained_tf_model'
path_to_prj_root = '../..'
model_file_corrected = str(path_to_prj_root / Path(model_file))
model = tf.keras.models.load_model(model_file_corrected)
model.summary()


@app.get('/')
async def root():
    return {'greeting': 'Howdy! This is the home page of the API for sunspots prediction.'}


@app.get('/prediction/')
async def prediction(sample: str = Query()):
    print(f'sample is {sample}\nof type {type(sample)}')

    # Parameters
    window_size = 30
    batch_size = 32

    # Print the model summary

    x = sample.split(',')
    assert len(x) >= window_size
    x = np.array([float(item) for item in x])
    x = x[-window_size:]
    y_pred = model_forecast(model, x, window_size, batch_size)
    results = y_pred.squeeze()
    return {'forecast': float(results)}
