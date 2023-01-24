
import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np


# Hyperparams Bejing
test_size = 0.25   ##0.25
batch_size = 64  #128
target_col_name = "pm2.5"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2013, 1, 1)

dim_val = 64
n_heads = 1
n_decoder_layers = 2
n_encoder_layers = 2

dec_seq_len = 20 # length of input given to decoder  # it is not used anywhere.. It has to be samewith target(trg not trg_y) dim

enc_seq_len = 48 # length of input given to encoder # only required to create data: src,tgt,tgt_y

# The transformer encoder and decoder module do not require encoder sequence length or decoder sequence length.
# checked transformer_timeseries.py. these two arg: enc_seq_len,dec_seq_len is never used.***

# output_sequence_length = 20 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
output_sequence_length = 2 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 64
in_features_decoder_linear_layer = 64
max_seq_len = enc_seq_len
batch_first = False

# Define input variables
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

### pytorch data object checks
src, trg, trg_y= training_data.__getitem__(index=1)
src.shape,trg.shape,trg_y.shape

# print('src:',src)
# print('tgt:',trg)
# print('trg_y:',trg_y)

# Making dataloader
training_data = DataLoader(training_data, batch_size)

## Dataloader check
# i, batch = next(enumerate(training_data))
# src, trg, trg_y = batch
# src.shape,trg.shape,trg_y.shape

import torch.nn as nn
model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    )

criterion=nn.MSELoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss=[]


for i, batch in enumerate(training_data):
    print(i)
    src, trg, trg_y = batch
    print(src.shape,trg.shape,trg_y.shape)

    src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    )
    
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.generate_square_subsequent_mask( 
        dim1=output_sequence_length,
        dim2=output_sequence_length
        )
    
    batch_first = False
    src = src.permute(1, 0, 2)
    trg = trg.permute(1, 0, 2)
    trg_y=trg_y.permute(1,0)
    
    print('afer permute new shape:',src.shape,trg.shape,trg_y.shape)

    output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )

    output=output[:,:,-1]
    print ('output shape:',output.shape)

    print('src:',src)
    print('tgt:',trg)
    print('trg_y:',trg_y)
    print('output:',output)

    optimizer.zero_grad()
    
    loss=criterion(output,trg_y)
    print('loss:',loss)
    loss.backward()
    # Adjust learning weights
    optimizer.step()



PATH='~/model.pt'


torch.save(model, PATH)
