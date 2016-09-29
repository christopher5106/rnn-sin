require 'rnn'
require 'gnuplot'
require 'nngraph'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a model to predict sin')
cmd:text()
cmd:text('Options')
cmd:option('-model', 'lstm', 'lstm, grid-lstm')
cmd:option('-select', 'true', 'use a selecttable')
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-dropout',0,'dropout for regularization in grid-lstm. 0 = no dropout')
cmd:option('-nb_layers',2,'Number of layers')
cmd:option('-rho',50,'Sequence length')
cmd:option('-batch_size',100,'Batch size')
cmd:option('-hidden_size',300,'Hidden layer size')
cmd:option('-iters',10000,'number of iterations')
cmd:option('-gpu',1,'GPU to use. 0 = no GPU')
cmd:option('-resume','','Resume from a model')
cmd:option('-eval_every',1500,'Evaluation every number of iterations')
cmd:option('-nPredict',152,'Number of predictions during testing')
cmd:option('-eval_one_step','false','Eval only one step next during testing')
cmd:option('-hypertuning','false','Parameter hypertuning')
opt = cmd:parse(arg)


if opt.gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end

nIndex = 1

local rnn

if opt.resume ~= '' then
  rnn = torch.load(opt.resume)
else

  if opt.model == 'lstm' then

    rnn = nn.Sequential()
       :add(nn.FastLSTM(nIndex, opt.hidden_size))
       :add(nn.NormStabilizer())

    for i = 2,opt.nb_layers do
       rnn:add(nn.FastLSTM(opt.hidden_size, opt.hidden_size))
       :add(nn.NormStabilizer())
    end

  elseif opt.model == 'grid-lstm' then

    rnn = nn.Sequential()
      :add(nn.Linear(nIndex, opt.hidden_size))
      :add(nn.Grid2DLSTM(opt.hidden_size, opt.nb_layers, opt.dropout, true, opt.rho))

  end

-- just 2 different ways to write:
  if opt.select ~= 'false' then
    print("Mode selecttable")
    rnn = nn.Sequential()
        :add(nn.SplitTable(0,2))
        :add(nn.Sequencer(rnn))
        :add(nn.SelectTable(-1))
        :add(nn.Dropout(0.5))
        :add(nn.Linear(opt.hidden_size, opt.hidden_size))
        :add(nn.ReLU())
        :add(nn.Linear(opt.hidden_size, nIndex))
        :add(nn.HardTanh())

  else
    print("Mode without selecttable")
    rnn:add(nn.Dropout(0.5))
     :add(nn.Linear(opt.hidden_size, opt.hidden_size))
     :add(nn.ReLU())
     :add(nn.Linear(opt.hidden_size, nIndex))
     :add(nn.HardTanh())

    rnn = nn.Sequencer(rnn)

  end

end

rnn:training()
print(rnn)

if opt.gpu>0 then
  rnn=rnn:cuda()
end

criterion = nn.MSECriterion()
if opt.gpu>0 then
  criterion=criterion:cuda()
end

ii=torch.linspace(0,20000, 200000)
sequence=torch.cos(ii)
if opt.gpu>0 then
  sequence=sequence:cuda()
end

function pt (a)
  for k, v in pairs( a ) do
    print(k, v)
  end
end

function eval(i)
  rnn:evaluate()
  local predict = torch.FloatTensor(opt.nPredict)
  if opt.gpu>0 then
    predict=predict:cuda()
  end
  for step=1,opt.rho do
    predict[step]= sequence[step]
  end

  local start
  if opt.select ~= 'false' then
    start=torch.Tensor(opt.rho,1,nIndex):zero()
    if opt.gpu>0 then
      start=start:cuda()
    end
  else
    start = {}
  end

  for iteration=0,(opt.nPredict-opt.rho-1) do
    for step=1,opt.rho do
      start[step] = predict:index(1,torch.LongTensor({step+iteration})):view(1,1)
    end
    rnn:forget()
    local output = rnn:forward(start)

    if opt.eval_one_step ~= 'false' then
      predict[iteration+opt.rho+1] = sequence[iteration + opt.rho +1]
    else
      if opt.select ~= 'false' then
        predict[iteration+opt.rho+1] = output
      else
        predict[iteration+opt.rho+1] = (output[opt.rho]:float())[1][1]
      end
    end
  end

  gnuplot.pngfigure("output_" .. opt.nb_layers .. "x" .. opt.model .. "_b" .. opt.batch_size .. "_h" .. opt.hidden_size .. "_i" .. i .. ".png")
  gnuplot.plot({'predict',predict,'+'},{'sinus',sequence:narrow(1,1,opt.nPredict),'-'})
  gnuplot.plotflush()

end

function train()

  offsets = {}
  for i=1,opt.batch_size do
     table.insert(offsets, math.ceil(math.random()* (sequence:size(1)-opt.rho) ))
  end
  offsets = torch.LongTensor(offsets)
  if opt.gpu>0 then
    offsets=offsets:cuda()
  end

  local gradOutputsZeroed = {}
  for step=1,opt.rho do
    gradOutputsZeroed[step] = torch.zeros(opt.batch_size,1)
    if opt.gpu>0 then
      gradOutputsZeroed[step] = gradOutputsZeroed[step]:cuda()
    end
  end

  local train_losses = {}

  for iteration=1,opt.iters do
    rnn:forget()
    local inputs, targets
    if opt.select ~= 'false' then
      inputs=torch.Tensor(opt.rho,opt.batch_size,nIndex):zero()
      if opt.gpu>0 then
        inputs=inputs:cuda()
      end
    else
      inputs = {}
    end
    for j=1,opt.batch_size do
       if offsets[j] > sequence:size(1) - opt.rho then
          offsets[j] = 1
       end
    end
    for step=1,opt.rho do
        inputs[step] = sequence:index(1, offsets):view(opt.batch_size,nIndex)
        offsets:add(1)
    end
    targets = sequence:index(1, offsets)
    rnn:zeroGradParameters()
    local outputs = rnn:forward(inputs)

    local err
    if opt.select ~= 'false' then
      err = criterion:forward(outputs, targets)
      local gradOutputs = criterion:backward(outputs, targets)
      local gradInputs = rnn:backward(inputs, gradOutputs)
    else
      err = criterion:forward(outputs[opt.rho], targets)
      local gradOutputs = criterion:backward(outputs[opt.rho], targets)
      gradOutputsZeroed[opt.rho] = gradOutputs
      local gradInputs = rnn:backward(inputs, gradOutputsZeroed)
    end
    print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
    train_losses[iteration] = err
    rnn:updateParameters(opt.learning_rate)

     if iteration % opt.eval_every == 0 or iteration == opt.iters then
       eval(iteration)
      --  torch.save("model_" .. iteration .. ".t7", rnn)
       opt.learning_rate = opt.learning_rate * 0.1
     end
  end
  torch.save("train_losses_" .. opt.nb_layers .. "x" .. opt.model .. "_b" .. opt.batch_size .. "_h" .. opt.hidden_size .. "_i" .. opt.iters .. ".t7",train_losses)
  return train_losses
  -- torch.save("model.t7", rnn)
end

if opt.hypertuning ~= 'false' then
  print("Hypertuning")
  for n = 1,6 do
    for _,m in ipairs{"lstm","grid-lstm"} do
      for _,b in ipairs{1, 10, 100, 500 } do
        for _,h in ipairs{10, 100, 200, 300} do
          for _,d in ipairs{0,0.2,0.5} do
            opt.nb_layers = n
            opt.model = m
            opt.batch_size = b
            opt.hidden_size = h
            opt.dropout = d
            opt.iters = 4000 -- math.ceil( 400001 / b)
            opt.eval_every = 1000 -- math.ceil(50000 / b)
            print(opt.nb_layers, opt.model, "batch size: " .. opt.batch_size, "hidden size: " .. opt.hidden_size,"dropout: " .. opt.dropout)
            train()
          end
        end
      end
    end
  end
else
  train()
end
