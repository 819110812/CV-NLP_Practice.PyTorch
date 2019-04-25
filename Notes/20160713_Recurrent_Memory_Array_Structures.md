# [Recurrent Memory Array Structures](http://arxiv.org/abs/1607.03085), [code](https://github.com/krocki/ArrayLSTM)

## Basic LSTM

$$
\begin{cases}
f^t=\sigma(W_fx^t+U_fh^{t-1}+b_f)\\
i^t=\sigma(W_ix^t+U_ih^{t-1}+b_i)\\
o^t=\sigma(W_o^t+U_oh^{t-1}+b_o)\\
\hat{c}^{t}=tanh(W_cx^t+U_ch^{t-1}+b_c)\\
c^t=f_t\odot c^{t-1}+i_t\odot \hat{c}^{t}
\end{cases}
$$

```lua

local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

```

## State-sharing Memory: Array-LSTM

$$
\begin{cases}
f_k^t=\sigma(W_{kf}x^t+U_{kf}h^{t-1}+b_{kf})\\
i_k^t=\sigma(W_{ki}x^t+U_{ki}h^{t-1}+b_{ki})\\
o_k^t=\sigma(W_{ko}^t+U_{ok}h^{t-1}+b_{ko})\\
\hat{c}_k^{t}=tanh(W_{kc}x^t+U_{kc}h^{t-1}+b_{kc})\\
c_k^t=f_k^t\odot c_k^{t-1}+i_k^t\odot \hat{c}_k^{t} \\
h^t=\sum_ko_k^t\odot tanh(c_k^t)
\end{cases}
$$

```lua
function ArrayLSTM.Arraylstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    for S = 1,4 do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h   = inputs[(L-1)*5+6]
    local prev_c = {}
    for C = 1,4 do
      table.insert(prev_c, inputs[(L-1)*5+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*5]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local next_c = {}
    local next_h = {}
    for C =1,4 do
      local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1)
      local forget_gate = nn.Sigmoid()(n2)
      local out_gate = nn.Sigmoid()(n3)
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end
    local next_h_sum = next_h[1]
    for C = 2,4 do
      next_h_sum = nn.CAddTable()({next_h_sum, next_h[C]})
    end

    for C = 1,4 do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
```

## Deterministic Array-LSTM extensions(Lane selection: Soft attention)




$$
a_k^t=s_\sigma(W_{ka}x^t+U_{ka}h^{t-1}+b_{ka})\rightarrow s_k^t=\frac{e^{a_k^t}}{\sum_ke^{a_k^t}}\Leftrightarrow
\begin{cases}
f_k^t=s_k^t\odot \sigma(W_{kf}x^t+U_{kf}h^{t-1}+b_{kf})\\
f_k^t=s_k^t\odot \sigma(W_{kf}x^t+U_{kf}h^{t-1}+b_{kf})\\
i_k^t=s_k^t\odot \sigma(W_{ki}x^t+U_{ki}h^{t-1}+b_{ki})\\
o_k^t=s_k^t\odot \sigma(W_{ko}^t+U_{ok}h^{t-1}+b_{ko})\\
\hat{c}_k^{t}=tanh(W_{kc}x^t+U_{kc}h^{t-1}+b_{kc})\\
c_k^t=(1-f_k^t)\odot c_k^{t-1}+i_k^t\odot \hat{c}_k^{t} \\
h^t=\sum_ko_k^t\odot tanh(c_k^t)
\end{cases}
$$


```lua
function ArrayLSTM.ArraylstmSoftAtten(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    for S = 1,4 do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h   = inputs[(L-1)*5+6]
    local prev_c = {}
    for C = 1,4 do
      table.insert(prev_c, inputs[(L-1)*5+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*5]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local attention = {}
    local n1 = {}
    local n2 = {}
    local n3 = {}
    local n4 = {}

    for C =1,4 do
      local i2h =  nn.Linear(input_size_L, 5 * rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 5 * rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(5, rnn_size)(all_input_sums)
      local n1_tmp, n2_tmp, n3_tmp, n4_tmp, n5_tmp = nn.SplitTable(2)(reshaped):split(5)
      table.insert(n1, nn.Sigmoid()(n1_tmp))
      table.insert(n2, nn.Sigmoid()(n2_tmp))
      table.insert(n3, nn.Sigmoid()(n3_tmp))
      table.insert(n4, nn.Tanh()(n4_tmp))
      -- attention signals
      table.insert(attention, nn.Sigmoid()(n5_tmp))
    end

    local attention_sum = nn.Exp()(attention[1])
    for C =2,4 do
      attention_sum = nn.CAddTable()({attention_sum, nn.Exp()(attention[C])})
    end

    local next_c = {}
    local next_h = {}
    for C =1,4 do
      local attention_norm = nn.CDivTable()({nn.Exp()(attention[C]), attention_sum})
      local in_gate = nn.CMulTable()({n1[C],attention_norm})
      local forget_gate = nn.CMulTable()({n2[C],attention_norm})
      local out_gate = nn.CMulTable()({n3[C],attention_norm})
      local in_transform = n4[C]
      -- perform the LSTM update
      --table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({nn.AddConstant(1,true)(nn.MulConstant(-1,true)(forget_gate)), prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end

    local next_h_sum = next_h[1]
    for C = 2,4 do
      next_h_sum = nn.CAddTable()({next_h_sum, next_h[C]})
    end

    for C = 1,4 do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
```

## Non-deterministic Array-LSTM extensions

### Stochastic Output Pooling

```lua
function ArrayLSTM.ArraylstmStochasticPooling(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    for S = 1,4 do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h   = inputs[(L-1)*5+6]
    local prev_c = {}
    for C = 1,4 do
      table.insert(prev_c, inputs[(L-1)*5+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*5]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local n1 = {}
    local n2 = {}
    local n3 = {}
    local n4 = {}

    for C =1,4 do
      local i2h =  nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
      local n1_tmp, n2_tmp, n3_tmp, n4_tmp = nn.SplitTable(2)(reshaped):split(4)
      -- only has 4
      table.insert(n1, nn.Sigmoid()(n1_tmp))
      table.insert(n2, nn.Sigmoid()(n2_tmp))
      table.insert(n3, nn.Sigmoid()(n3_tmp))
      table.insert(n4, nn.Tanh()(n4_tmp))
    end

    local output_gates = nn.ConcatTable()({nn.ConcatTable()({n2[1],n2[2]}),nn.ConcatTable()({n2[3],n2[4]})})
    local output_proj = nn.Linear(4 * rnn_size, 4)(n2):annotate{name='stochasticdecoder'}
    local output_gates_soft = nn.Sigmoid()(nn.LogSoftMax()(output_proj))

    local next_c = {}
    local next_h = {}
    for C =1,4 do
      -- perform the LSTM update
      local out_gate = n2[C]
      local in_gate = n1[C]
      local forget_gate =n3[C]
      local in_transform = n4[C]
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end

    local next_h_sum = nn.CMulTable()({next_h[1],output_gates_soft[1]})
    for C = 2,4 do
      next_h_sum = nn.CAddTable()({next_h_sum, nn.CMul(output_gates_soft[C])(next_h[C])})
    end
    for C = 1,4 do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return ArrayLSTM
```

### Stochastic Memory Array
