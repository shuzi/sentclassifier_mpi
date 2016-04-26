--  CNN Train and Testing
--  Author: Minwei Feng (mfeng@us.ibm.com)

local opt = opt or {}
local state = {}
local conf = conf or {}
require 'sys'
tm = {}
tm.sync = 0
tm.fprop = 0
tm.transfer = 0
tm.feval = 0
tm.bprop = 0
tm.cbprop = 0
tm.err = 0
tm.conf = 0
tm.params = 0
tm.loss = 0
tm.test = 0
local ffi = require "ffi"
ffi.cdef "unsigned int sleep(unsigned int seconds);"

if opt.validMode == 'additionalTester' and conf.tranks[conf.rank] == true then
  print('Client ' .. tostring(conf.rank) ..' ready to run testing')
else
  print('Client ' .. tostring(conf.rank) ..' ready to run training')
end

model = nn.Sequential()
model:add(nn.LookupTable(mapWordIdx2Vector:size()[1], opt.embeddingDim))
model:add(nn.View(opt.batchSize*trainDataTensor:size()[2], opt.embeddingDim))
model:add(nn.Linear(opt.embeddingDim, opt.wordHiddenDim))
model:add(nn.View(opt.batchSize, trainDataTensor:size()[2], opt.wordHiddenDim))
model:add(nn.Tanh())
model:add(nn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
model:add(nn.Max(2))
model:add(nn.Tanh())
model:add(nn.Linear(opt.numFilters, opt.hiddenDim))
model:add(nn.Tanh())
model:add(nn.Linear(opt.hiddenDim, opt.numLabels))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

model:get(1).weight:copy(mapWordIdx2Vector)

model_test = nn.Sequential()
model_test:add(nn.LookupTable(mapWordIdx2Vector:size()[1], opt.embeddingDim))
model_test:add(nn.View(opt.batchSizeTest*validDataTensor:size()[2], opt.embeddingDim))
model_test:add(nn.Linear(opt.embeddingDim, opt.wordHiddenDim))
model_test:add(nn.View(opt.batchSizeTest, validDataTensor:size()[2], opt.wordHiddenDim))
model_test:add(nn.Tanh())
model_test:add(nn.TemporalConvolution(opt.wordHiddenDim, opt.numFilters, opt.contConvWidth))
model_test:add(nn.Max(2))
model_test:add(nn.Tanh())
model_test:add(nn.Linear(opt.numFilters, opt.hiddenDim))
model_test:add(nn.Tanh())
model_test:add(nn.Linear(opt.hiddenDim, opt.numLabels))
model_test:add(nn.LogSoftMax())

model_test:get(1).weight = model:get(1).weight
model_test:get(3).weight = model:get(3).weight
model_test:get(3).bias = model:get(3).bias
model_test:get(6).weight = model:get(6).weight
model_test:get(6).bias = model:get(6).bias
model_test:get(9).weight = model:get(9).weight
model_test:get(9).bias = model:get(9).bias
model_test:get(11).weight = model:get(11).weight
model_test:get(11).bias = model:get(11).bias


if opt.type == 'cuda' and conf.tranks[conf.rank] ~= true then
   model:cuda()
   criterion:cuda()
   model_test:cuda()
   model_test:get(1).weight = model:get(1).weight
   model_test:get(3).weight = model:get(3).weight
   model_test:get(3).bias = model:get(3).bias
   model_test:get(6).weight = model:get(6).weight
   model_test:get(6).bias = model:get(6).bias
   model_test:get(9).weight = model:get(9).weight
   model_test:get(9).bias = model:get(9).bias
   model_test:get(11).weight = model:get(11).weight
   model_test:get(11).bias = model:get(11).bias
end
if model then
   parameters,gradParameters = model:getParameters()
--   parameters = parameters:contiguous()
--   gradParameters = gradParameters:contiguous()
   parametersClone = parameters:clone()
end

if conf.rank == 0 then
   print(model)
   print(criterion)
   print("Model Size: ", parameters:size()[1])
end



-------------------------------------------------------------------
require 'optim'
local opti
if opt.optimization == 'sgd' then
   opti = optim.msgd
   state.optconf = {
      lr = opt.learningRate,
      lrd = opt.weightDecay,
      mom = opt.momentum,
      pclient = pc    
   }
elseif opt.optimization == 'downpour' then
   opti = optim.downpour
   state.optconf = {
      lr = opt.learningRate,
      lrd = opt.weightDecay,
      pclient = pc,
      su = opt.commperiod    
   }
elseif opt.optimization == 'eamsgd' then
   opti = optim.eamsgd
   state.optconf = {
      lr = opt.learningRate,
      lrd = opt.weightDecay,
      pclient = pc,
      su = opt.commperiod,
      mva = opt.movingrate,
      mom = opt.momentum
   }
elseif opt.optimization == 'rmsprop' then
   opti = optim.rmsprop
   state.optconf = {
      mode = opt.modeRMSProp,
      decay = opt.decayRMSProp,
      lr = opt.lrRMSProp,
      momentum = opt.momentumRMSProp,
      epsilon = opt.epsilonRMSProp,
      pclient = pc,
      su = opt.commperiod      
   }
elseif opt.optimization == 'rmspropsingle' then
   opti = optim.rmspropsingle
   state.optconf = {
      decay = opt.decayRMSProp,
      lr = opt.lrRMSProp,
      momentum = opt.momentumRMSProp,
      epsilon = opt.epsilonRMSProp,
      pclient = pc   
   }
elseif opt.optimization == 'adam' then
   opti = optim.adam
   state.optconf = {
      mode = opt.modeAdam,
      lr = opt.lrAdam,
      beta1 = opt.beta1Adam,
      beta2 = opt.beta2Adam,
      epsilon = opt.epsilonAdam,
      pclient = pc,
      su = opt.commperiod      
   }
elseif opt.optimization == 'adamsingle' then
   opti = optim.adamsingle
   state.optconf = {
      lr = opt.lrAdam,
      beta1 = opt.beta1Adam,
      beta2 = opt.beta2Adam,
      epsilon = opt.epsilonAdam,
      pclient = pc     
   }
elseif opt.optimization == 'adamax' then
   opti = optim.adamax
   state.optconf = {
      mode = opt.modeAdam,
      lr = opt.lrAdam,
      beta1 = opt.beta1Adam,
      beta2 = opt.beta2Adam,
      epsilon = opt.epsilonAdam,
      pclient = pc,
      su = opt.commperiod      
   }   
elseif opt.optimization == 'adamaxsingle' then
   opti = optim.adamaxsingle
   state.optconf = {
      lr = opt.lrAdam,
      beta1 = opt.beta1Adam,
      beta2 = opt.beta2Adam,
      epsilon = opt.epsilonAdam,
      pclient = pc     
   }
elseif opt.optimization == 'adagrad' then
   opti = optim.adagrad
   state.optconf = {
      mode = opt.modeAdagrad,
      lr = opt.lrAdagrad,
      lrd = opt.lrDecayAdagrad,
      epsilon = opt.epsilonAdagrad,
      pclient = pc,
      su = opt.commperiod          
   }   
elseif opt.optimization == 'adagradsingle' then
   opti = optim.adagradsingle
   state.optconf = {
      lr = opt.lrAdagrad,
      lrd = opt.lrDecayAdagrad,
      epsilon = opt.epsilonAdagrad,
      pclient = pc
   }   
elseif opt.optimization == 'adadelta' then
   opti = optim.adadelta
   state.optconf = {
      mode = opt.modeAdadelta,
      rho = opt.rhoAdadelta,
      lr = opt.lrAdadelta,
      epsilon = opt.epsilonAdadelta,
      pclient = pc,
      su = opt.commperiod       
   } 
elseif opt.optimization == 'adadeltasingle' then
   opti = optim.adadeltasingle
   state.optconf = {
      rho = opt.rhoAdadelta,
      epsilon = opt.epsilonAdadelta,
      lr = opt.lrAdadelta,
      pclient = pc     
   } 
else
   os.error('unknown optimization method')
end


local pclient = pc
if pclient then
   pclient:start(parameters,gradParameters)
end
 

local input = nil
local target = nil
local feval = function(x)
   if x ~= parameters then
      parameters:copy(x)
   end
   gradParameters:zero()
   local f = 0
   local output = model:forward(input)
   f = criterion:forward(output, target)
   local df_do = criterion:backward(output, target)
   model:backward(input, df_do)

   if opt.L1reg ~= 0 then
      local norm, sign = torch.norm, torch.sign
      f = f + opt.L1reg * norm(parameters,1)
      gradParameters:add( sign(parameters):mul(opt.L1reg) )
   end
   if opt.L2reg ~= 0 then
      local norm, sign = torch.norm, torch.sign
      f = f + opt.L2reg * norm(parameters,2)^2/2
      parametersClone:copy(parameters)
      gradParameters:add( parametersClone:mul(opt.L2reg) )
   end
 --  gradParameters:clamp(-opt.gradClip, opt.gradClip)

   return f,gradParameters
end

function test(inputDataTensor, inputTarget, state)
    local time = sys.clock()
    model_test:evaluate()
    local bs = opt.batchSizeTest
    local batches = inputDataTensor:size()[1]/bs
    local correct = 0
    for t = 1,batches,1 do
        local begin = (t - 1)*bs + 1
        local input = inputDataTensor:narrow(1, begin , bs)

        local pred = model_test:forward(input)
        local prob, pos = torch.max(pred, 2)
        for m = 1,bs do
           for k,v in ipairs(inputTarget[begin+m-1]) do
            if pos[m][1] == v then
                correct = correct + 1
                break
            end
          end
        end
    end
    state.bestAccuracy = state.bestAccuracy or 0
    local currAccuracy = correct/(inputDataTensor:size()[1])
    if currAccuracy > state.bestAccuracy then state.bestAccuracy = currAccuracy; end
    print(string.format("Accuracy: %s, best Accuracy: %s  at time %s", currAccuracy, state.bestAccuracy, sys.toc() ))
    return currAccuracy
end




-- train
sys.tic()
avg_err = 0
iter = 0
pversion = 0

if opt.validMode == 'additionalTester' and conf.tranks[conf.rank] == true then
    local trainState = {}
    local validState = {}
    local testState = {}
    validState.prevacc = -1
    local countsame = 0
    while true do
      ffi.C.sleep(opt.validSleepTime)
      local comm_time_4test = sys.clock()
      print(string.format("Client %s: before receive", conf.rank))
      pclient:async_recv_param()
      pclient:wait()
      print(string.format("Client %s: communication time: %.2f ",
                          conf.rank, sys.clock() - comm_time_4test))
 --     test(trainDataTensor, trainDataTensor_y_2, trainState)
      local acc = test(validDataTensor, validDataTensor_y, validState)
      if acc == validState.prevacc then
         countsame = countsame + 1
      else
         countsame = 0
         validState.prevacc = acc
      end
      test(testDataTensor, testDataTensor_y, testState)
      
      if opt.outputprefix ~= 'none' then
         torch.save(opt.outputprefix ..
            string.format("_%010.2f_model",
                sys.toc()+opt.prevtime), parameters)
      end
      
      if countsame > 5 then
         break
      end
   end
else
    for epoch = 1,opt.epoch do
        local time_epoch = sys.clock()
        model:training()
        local batches = trainDataTensor:size()[1]/opt.batchSize
        local bs = opt.batchSize
        
   --     local shuffle = torch.FloatTensor(batches)
   --     local shuffle = torch.randperm(batches)
   --     torch.setdefaulttensortype('torch.CudaTensor')
        local shuffle = torch.ones(batches)
        if batches > 1 then
          for i = 1,batches do
            shuffle[i] = i
          end
          for i = batches,2,-1 do
            local j = math.random(1,i)
            local temp = shuffle[i]
            shuffle[i] = shuffle[j]
            shuffle[j] = temp
          end
        end
        
        local cost =  0
        for t = 1,batches,1 do
            local begin = (shuffle[t] - 1)*bs + 1
          --  print('client ' .. tostring(conf.rank) .. ' : begin batch ' .. tostring(begin) )
            input = trainDataTensor:narrow(1, begin , bs) 
            target = trainDataTensor_y:narrow(1, begin , bs)
           
            local c,g = opti(feval, parameters, state.optconf)
            cost = cost + c[1]
            model:get(1).weight:narrow(1,1,2):fill(0)
        end
        print('client ' .. tostring(conf.rank) .. ':' .. ' epoch ' .. epoch .. ' done, avg. loss ' .. tostring(cost/batches)  
            .. ' for ' .. (sys.clock() - time_epoch) .. ' seconds ')
    end
end

if pclient then
   pclient:stop()
end

print('Client ' .. conf.rank .. ' total training time is ' .. sys.toc())
if state.optconf.dusync then
   tm.sync = state.optconf.dusync
end
print('Client ' .. conf.rank .. ' total sync time is ' .. tm.sync)
