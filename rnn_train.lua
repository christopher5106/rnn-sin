require 'rnn'
require 'gnuplot'

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

nIters = 10000
batchSize = 100
rho = 50
hiddenSize = 300
nIndex = 1
lr = 0.01


-- rm = nn.Sequential() -- input is {x[t], h[t-1]}
--  :add(nn.ParallelTable()
--     :add(nn.Linear(nIndex, hiddenSize)) -- input layer
--     :add(nn.Linear(hiddenSize, hiddenSize))) -- recurrent layer
--  :add(nn.CAddTable()) -- merge
--  :add(nn.Sigmoid()) -- transfer
--
-- rnn = nn.Sequential()
--  :add(nn.Recurrence(rm, hiddenSize, 1, rho))

rnn = nn.Sequential()
   :add(nn.FastLSTM(nIndex, hiddenSize))
   :add(nn.NormStabilizer())
   :add(nn.FastLSTM(hiddenSize, hiddenSize))
   :add(nn.NormStabilizer())
   :add(nn.Dropout(0.5))
   :add(nn.Linear(hiddenSize, hiddenSize))
   :add(nn.ReLU())
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.HardTanh())

rnn = nn.Sequencer(rnn)


rnn:training()
print(rnn)

if gpu>0 then
  rnn=rnn:cuda()
end

criterion = nn.MSECriterion()
if gpu>0 then
  criterion=criterion:cuda()
end

ii=torch.linspace(0,20000, 200000)
sequence=torch.cos(ii)
if gpu>0 then
  sequence=sequence:cuda()
end

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()* (sequence:size(1)-rho) ))
end
offsets = torch.LongTensor(offsets)
if gpu>0 then
  offsets=offsets:cuda()
end

local gradOutputsZeroed = {}
for step=1,rho do
  gradOutputsZeroed[step] = torch.zeros(batchSize,1)
  if gpu>0 then
    gradOutputsZeroed[step] = gradOutputsZeroed[step]:cuda()
  end
end

for iteration=1,nIters do
  rnn:forget()
   local inputs, targets = {}, {}
   for step=1,rho do
      inputs[step] = sequence:index(1, offsets):view(batchSize,nIndex)
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > sequence:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
   end
   rnn:zeroGradParameters()
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs[rho], targets[rho])
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   local gradOutputs = criterion:backward(outputs[rho], targets[rho])
   gradOutputsZeroed[rho] = gradOutputs
   local gradInputs = rnn:backward(inputs, gradOutputsZeroed)
   rnn:updateParameters(lr)

   if iteration % 1500 == 0 then
     torch.save("model_" .. iteration .. ".t7", rnn)
     lr = lr * 0.1
   end
end

torch.save("model.t7", rnn)
