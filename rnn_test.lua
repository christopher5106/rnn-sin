function pt (a)
  for k, v in pairs( a ) do
    print(k, v)
  end
end

require 'rnn'
require 'gnuplot'
gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

rnn = torch.load("model_1500.t7")
nPredict=152
rho = 50
nIndex = 1

ii=torch.linspace(0,20000, 200000)
sequence=torch.cos(ii)
if gpu>0 then
  sequence=sequence:cuda()
end

rnn:evaluate()
predict=torch.FloatTensor(nPredict)
if gpu>0 then
  predict=predict:cuda()
end
for step=1,rho do
  predict[step]= sequence[step]
end

start = {}

-- start=torch.Tensor(rho,1,nIndex):zero()
-- if gpu>0 then
--   start=start:cuda()
-- end

for iteration=0,(nPredict-rho-1) do
  for step=1,rho do
    start[step] = predict:index(1,torch.LongTensor({step+iteration})):view(1,1)
  end
  rnn:forget()
  output = rnn:forward(start)

  predict[iteration+rho+1] = (output[rho]:float())[1][1]
  -- predict[iteration+rho+1] = sequence[iteration + rho +1]
  -- predict[iteration+rho+1] = output
end

gnuplot.pngfigure("output.png")
gnuplot.plot({'predict',predict,'+'},{'sinus',sequence:narrow(1,1,nPredict),'-'})
gnuplot.plotflush()
