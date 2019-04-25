# [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623v2.pdf)

<!-- ![](https://www.dropbox.com/s/uka203ih85c2kn8/draw.png?dl=1) -->

## model define

1. reader

```lua
x = nn.Identity()()
x_error_prev = nn.Identity()()
read_module = READ.create(x, x_error_prev, opt.rnnSize, opt.sizeImage, opt.attenReadSize, opt.batchSize)
```

2. reader-lstm

```lua
input = nn.Identity()()
lstm_enc = LSTM.create(input, 2 * opt.attenReadSize * opt.attenReadSize, opt.rnnSize)
```

3. QSampler

```lua
next_h = nn.Identity()()
qsampler = QSampler.create(opt.rnnSize, next_h, opt.sizeLayerZ)
--
encoder = {read_module, lstm_enc, qsampler}
```

4. decoder

```lua
input = nn.Identity()()
lstn_dec = LSTM.create(input, opt.sizeLayerZ, opt.rnnSize)
```


5. writer

```lua
next_h = nn.Identity()()
prev_canvas = nn.Identity()()
write_module = WRITE.create(next_h, prev_canvas, opt.rnnSize, opt.sizeImage, opt.attenWriteSize, opt.batchSize)
```


6. loss
```lua
x = nn.Identity()()
next_canvas = nn.Identity()()
loss_x = LOSS_X.create(x,next_canvas)
--
decoder = {lstn_dec, write_module, loss_x}
```


## dataset

```lua
trainset = mnist.traindataset()
```

## train

1. get parameters(parameters and gradient parameters)
```lua
params, grad_params = model_utils.combine_all_parameters(encoder[1], encoder[2], encoder[3], decoder[1], decoder[2], decoder[3])
```


2. training loop(forward)

- get inputs and targets

```lua
inputs = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)
for i = 1, v:size(1) do
    inputs[{{i}, {}, {}}] = trainset[v[i]].x:gt(125):cuda() -- input training data
    targets[i] =  trainset[v[i]].y
end
```


- forward the model

```lua
for t = 1, opt.seqSize do
    ...
    loss = loss + combine_loss
end
loss = loss / opt.seqSize
```

```lua
e[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
x[t] = inputs
```
$$
\begin{cases}
x\\
\hat{x}_t=x-\sigma(c_{t-1})\\
r_t=\text{read}(x_t,\hat{x}_{t-1},h_{t-1}^{dec})
\end{cases}
$$
```lua
--encoder
patch[t], read_input[t]         = unpack(encoder_clones[t][1]:forward({x[t], x_error[t-1], lstm_h_dec[t-1], ascending}))
```
$$
h_t^{enc}=\text{RNN}^{enc}(h_{t-1}^{enc},[r_t,h_{t-1}^{dec}])
$$
```lua
lstm_c_enc[t], lstm_h_enc[t]    = unpack(encoder_clones[t][2]:forward({read_input[t],lstm_c_enc[t-1], lstm_h_enc[t-1]}))
```
$$\begin{cases}
z_t\sim Q(Z_t|h_t^{enc})\\
\mathcal{L}_z
\end{cases}
$$
```lua
z[t], loss_z[t]                 = unpack(encoder_clones[t][3]:forward({lstm_h_enc[t], e[t]}))
```
<b>loss_z is the first loss output.</b>
```lua
--decoder
lstm_c_dec[t], lstm_h_dec[t]          = unpack(decoder_clones[t][1]:forward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]}))
canvas[t]                             = decoder_clones[t][2]:forward({lstm_h_dec[t],ascending,canvas[t-1]})
loss_x[t],x_prediction[t],x_error[t]  = unpack(decoder_clones[t][3]:forward({canvas[t],x[t]}))
```
<b>loss_x is the final output</b>

$$
\mathcal{L}=\mathcal{L}_x+\mathcal{L}_z
$$
```lua
loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
```
<b>loss is the combine loss </b>



3. training loop(backward)

```lua
dlstm_c_enc = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
dlstm_h_enc = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
dlstm_c_dec = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
dlstm_h_dec = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
dlstm_h_dec2 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
dloss_z = {}
```


- backward Decoder

```lua
--decoder:backward(inputs, outputs)
(1) zero the accumulation of the gradients
dcanvas2 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
dx_error = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
dloss_x = {}
dloss_x[t] = torch.ones(opt.batchSize, 1)
dx_prediction[t] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)
dx_error = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
```
$$
\mathcal{L}_x=-\log D(x|c_T)\rightarrow \begin{cases}
\Delta_w \mathcal{L}_x(W,n;x,c_T)=\frac{\partial \mathcal{L}_x}{\partial w}=\frac{\partial \mathcal{L}_x}{\partial z}\frac{\partial z}{\partial w}\\
\Delta_b \mathcal{L}_x(W,n;x,c_T)=\frac{\partial \mathcal{L}_x}{\partial b}
\end{cases}
$$
```lua
(2) accumulate gradients
dcanvas2[t],dx1[t] = unpack(decoder_clones[t][3]:backward({canvas[t],x[t]},{dloss_x[t],dx_prediction[t],dx_error[t]}))
-- can not use single criterion:backward(mlp.output, output), because it has many error
```


```lua
(1) zero the accumulation of the gradients
dcanvas1 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
dcanvas1[t] = dcanvas1[t] + dcanvas2[t]
(2))merge gradient from canvas
dlstm_h_dec3 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
(3) accumulate gradients
dlstm_h_dec3[t],dascending1,dcanvas1[t-1] = unpack(decoder_clones[t][2]:backward({lstm_h_dec[t],ascending,canvas[t-1]},dcanvas1[t]))
```

```lua
(1) zero the accumulation of the gradients
dlstm_h_dec1 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
(2))merge gradient from lstm_h_dec1
dlstm_h_dec1[t] = (dlstm_h_dec1[t] + dlstm_h_dec3[t])
(3) accumulate gradients
dz[t], dlstm_c_dec[t-1], dlstm_h_dec1[t-1]  = unpack(decoder_clones[t][1]:backward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]},{dlstm_c_dec[t],dlstm_h_dec1[t] }))
```

```lua
-encoder
dlstm_h_enc[t], de[t] = unpack(encoder_clones[t][3]:backward({lstm_h_enc[t], e[t]},{dz[t],dloss_z[t]}))
```
```lua
dread_input[t], dlstm_c_enc[t-1], dlstm_h_enc[t-1] = unpack(encoder_clones[t][2]:backward({read_input[t],lstm_c_enc[t-1], lstm_h_enc[t-1]},{dlstm_c_enc[t], dlstm_h_enc[t]}))
```
```lua
dx2[t], dx_error[t-1], dlstm_h_dec2[t-1], dascending2 = unpack(encoder_clones[t][1]:backward({x[t], x_error[t-1], lstm_h_dec[t-1], ascending},{dpatch[t],dread_input[t]}))
```
```lua
--merge gradient from lstm_h_dec
dlstm_h_dec[t-1] = dlstm_h_dec1[t-1] + dlstm_h_dec2[t-1]
```
