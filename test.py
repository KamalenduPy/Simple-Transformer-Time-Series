"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np

# Hyperparams
test_size = 0.1
batch_size = 128
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)




#### Load Test data
gpu=False


test_data = data[-(round(len(data)*test_size)):]
test_data.shape
test_indices = utils.get_indices_entire_sequence(
    data=test_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
test_data = ds.TransformerDataset(
    data=torch.tensor(test_data[input_variables].values).float(),
    indices=test_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

test_dl = DataLoader(test_data, batch_size)



##### Check if the test loop working

# i,batch=next(enumerate(test_dl))
# src,trg,trg_y=batch
# print(src.shape,trg.shape,trg_y.shape)

# src = src.permute(1, 0, 2)
# trg = trg.permute(1, 0, 2)
# trg_y = trg_y.permute(1, 0)

# src_mask = utils.generate_square_subsequent_mask(
# dim1=output_sequence_length,
# dim2=enc_seq_len
# )
# tgt_mask = utils.generate_square_subsequent_mask( 
# dim1=output_sequence_length,
# dim2=output_sequence_length
# )

# yhat = model(src=src,tgt=trg,src_mask=src_mask,tgt_mask=tgt_mask)
# yhat=yhat[:,:,-1]
# print(trg_y.shape,yhat.shape)
# criterion=torch.nn.MSELoss()
# criterion(yhat,trg_y)


## Load Model
PATH='/content/gdrive/MyDrive/Colab_directory/TS_w_Transformer/bejing/model.pt'
model=torch.load(PATH)


### if Cuda is available
if gpu==True:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model=model.to(device)



###### function to evaluate the model


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    loss=list()
    for i, (src, trg, trg_y) in enumerate(test_dl):
        print(src.shape,trg.shape,trg_y.shape)
        # evaluate the model on the test set
        batch_first = False
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        trg_y = trg_y.permute(1, 0)
        if gpu==True:
            src=src.to(device)
            trg=trg.to(device)
            trg_y=trg_y.to(device)

        src_mask = utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
        )
        tgt_mask = utils.generate_square_subsequent_mask( 
        dim1=output_sequence_length,
        dim2=output_sequence_length
        )
        
        yhat = model(src=src,tgt=trg,src_mask=src_mask,tgt_mask=tgt_mask)
        yhat=yhat[:,:,-1]
        print(trg_y.shape,yhat.shape)

        criterion=torch.nn.MSELoss()
        l1= criterion(yhat,trg_y)
        print(l1)
        loss.append(l1)
    return(loss)

