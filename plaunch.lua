-- start point
-- Author: Minwei Feng (mfeng@us.ibm.com)

require 'mpiT'

mpiT.tag_ps_recv_init  = 1
mpiT.tag_ps_recv_grad  = 2
mpiT.tag_ps_send_param = 3
mpiT.tag_ps_recv_param = 4
mpiT.tag_ps_recv_header = 5
mpiT.tag_ps_recv_stop = 6
mpiT.tag_ps_recv_param_tail = 7
mpiT.tag_ps_recv_grad_tail = 8

dofile('init.lua')

cmd = torch.CmdLine('_')
cmd:text()
cmd:text('Options:')
cmd:option('-threads', 1, 'number of threads')
cmd:option('-optimization', 'downpour', 'optimization method: downpour | eamsgd | rmsprop ')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'training mini-batch size')
cmd:option('-batchSizeTest', 1, 'test mini-batch size')
cmd:option('-lrAdagrad', 1e-3, 'learning rate for adagrad')
cmd:option('-lrDecayAdagrad', 0.000001, 'learning rate decay for adagrad')
cmd:option('-epsilonAdagrad', 1e-10, 'epsilon for adagrad')
cmd:option('-modeAdagrad', 'global', 'mode for distributed adagrad, currently only global')
cmd:option('-rhoAdadelta', 0.9, 'rho for adadelta')
cmd:option('-lrAdadelta', 1, 'lr for adadelta')
cmd:option('-epsilonAdadelta', 1e-6, 'epsilon for adadelta')
cmd:option('-modeAdadelta', 'global', 'mode for distributed adadelta, currently only global')
cmd:option('-lrAdam', 1e-3, 'learning rate for adam')
cmd:option('-beta1Adam', 0.9, 'beta1 for adam')
cmd:option('-beta2Adam', 0.999, 'beta2 for adam')
cmd:option('-epsilonAdam', 1e-8, 'epsilon for adam')
cmd:option('-stepDivAdam', 72, 'step divide for adam')
cmd:option('-modeAdam', 'global', 'mode for adam, currently only global')
cmd:option('-gradClip', 0.5, 'boundary for gradient clip')
cmd:option('-weightDecay', 0.000001, 'weight decay')
cmd:option('-decayRMSProp', 0.95, 'decay for rmsprop')
cmd:option('-lrRMSProp', 1e-4, 'learning rate for rmsprop')
cmd:option('-momentumRMSProp', 0.9, 'momentum for rmsprop')
cmd:option('-epsilonRMSProp', 1e-4, 'epsilon for rmsprop')
cmd:option('-modeRMSProp', 'global', 'mode for rmsprop')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-commperiod' , 1, ' sync updates')
cmd:option('-movingrate' , 0.05, 'moving rate')
cmd:option('-type', 'float', 'type: float | cuda')
cmd:option('-trainFile', 'train', 'input training file')
cmd:option('-validFile', 'valid', 'input validation file')
cmd:option('-testFile', 'test', 'input test file')
cmd:option('-embeddingFile', 'embedding', 'input embedding file')
cmd:option('-embeddingDim', 100, 'input word embedding dimension')
cmd:option('-contConvWidth', 2, 'continuous convolution filter width')
cmd:option('-wordHiddenDim', 200, 'first hidden layer output dimension')
cmd:option('-numFilters', 1000, 'CNN filters amount')
cmd:option('-hiddenDim', 1000, 'second hidden layer output dimension')
cmd:option('-numLabels', 311, 'label quantity')
cmd:option('-epoch', 200, 'maximum epoch')
cmd:option('-L1reg', 0, 'L1 regularization coefficient')
cmd:option('-L2reg', 1e-4, 'L2 regularization coefficient')
cmd:option('-trainMaxLength', 150, 'maximum length for training')
cmd:option('-testMaxLength', 150, 'maximum length for valid/test')
cmd:option('-trainMinLength', 40, 'maximum length for training')
cmd:option('-testMinLength', 40, 'maximum length for valid/test')
cmd:option('-validMode', 'additionalTester', 'validation type: none | lastClient | additionalTester')
cmd:option('-validSleepTime', 300, 'validati4on sleep time in seconds, only for additionalTester')
cmd:option('-servRecvgrad', true, 'server recvgrad')
cmd:option('-servSendparam', true, 'server send param to workers')
cmd:option('-outputprefix', 'none', 'output file prefix')
cmd:option('-prevtime', 0, 'time start point')
cmd:option('-loadmodel', 'none', 'load model file name')
cmd:option('-preloadBinary', false, 'load data from binary files')
cmd:option('-testerfirst', false, 'rank 0 is the tester')
cmd:option('-testerlast', false, 'last rank is the tester')
cmd:option('-masterFreq', 2, 'this parameter control the ratio of master and client')
cmd:option('-maxrank', 16, 'max rank used')
cmd:text()
opt = cmd:parse(arg or {})


mpiT.Init()
local world = mpiT.COMM_WORLD
local rank = mpiT.get_rank(world)
local size = mpiT.get_size(world)

local oncuda 
local AGPU = nil
if opt.type == 'cuda' then
    oncuda = true
    require 'cutorch'
    require 'cunn'
    torch.manualSeed(rank)
    math.randomseed(rank)
    cutorch.manualSeed(rank)
    AGPU = {1,2}
else
    oncuda = false
    require 'nn'
    require 'nngraph'
    torch.manualSeed(rank)
    math.randomseed(rank)
end

size = opt.maxrank + 1
opt.rank = rank
if rank > opt.maxrank then
  print('rank ' .. rank .. ' do nothing ')
  while true do
    sys.usleep(1000)
  end
end

local gpu = nil
if rank == 1 then
  print(opt)
end
conf = {}
conf.lr = opt.learningRate
conf.rank = rank
conf.size = size
conf.world = world
conf.sranks = {}
conf.cranks = {}
conf.tranks = {}
conf.oncuda = oncuda
conf.opt = opt


if opt.validMode == 'additionalTester' then
   if size % 2 ~= 1 then
      error("validMode additionalTester requires size be an odd number")
   end
end

-- notice the rank starts from 0
local role = nil
if opt.testerfirst then
   table.insert(conf.cranks,0)
   if rank == 0 then
      role = 'pe'
   end
   for i = 1,size-1 do
      if math.fmod(i,opt.masterFreq) ~= 0 then
         table.insert(conf.cranks,i)
         if rank == i then
            role = 'pt'
         end
      else
         table.insert(conf.sranks,i)
         if rank == i then
            role = 'ps'
         end
      end
   end
end

if opt.testerlast then
   for i = 0,size-2 do
      if math.fmod(i+1,opt.masterFreq) ~= 0 then
         table.insert(conf.cranks,i)
         if rank == i then
            role = 'pt'
         end
      else
         table.insert(conf.sranks,i)
         if rank == i then
            role = 'ps'
         end
      end
   end
   table.insert(conf.cranks,size-1)
   if rank == size - 1 then
      role = 'pe'
   end
end


if opt.validMode == 'lastClient' then
    conf.tranks[size-1] = true
elseif opt.validMode == 'additionalTester' then
   -- set rank 0 as the additionalTester
    if opt.testerfirst then
       conf.tranks[0] = true
    elseif opt.testerlast then
       conf.tranks[size-1] = true
    else
       error("Incorrect configuration")
    end
elseif opt.validMode == 'none' then
    for i = 0,size-1 do
      if math.fmod(i+1,opt.masterFreq) ~= 0 then
         table.insert(conf.cranks,i)
         if rank == i then
            role = 'pt'
         end
      else
         table.insert(conf.sranks,i)
         if rank == i then
            role = 'ps'
         end
      end
   end
end

if role == 'ps' then
   -- server   
    print('[server] rank ' .. tostring(rank) .. ' use cpu' .. " on " .. tostring(io.popen('hostname -s'):read()))
    torch.setdefaulttensortype('torch.FloatTensor')
--   end
    local ps = pServer(conf)
    ps:start()
else
   if oncuda and role == 'pt' then
      local gpus = cutorch.getDeviceCount()
      local gpuid = AGPU[(rank/2) % gpus + 1]
      cutorch.setDevice(gpuid)
      print('[client] rank '.. tostring(rank) .. ' use gpu ' .. tostring(gpuid) .. " on " .. tostring(io.popen('hostname -s'):read()))
      torch.setdefaulttensortype('torch.CudaTensor')
      opt.gpuid = gpuid
   elseif not oncuda and role == 'pt' then
      print('[client] rank '.. tostring(rank) .. ' use cpu ' .. " on " .. tostring(io.popen('hostname -s'):read()))
      torch.setdefaulttensortype('torch.FloatTensor')
   else
      print('[tester] rank '.. tostring(rank) .. ' use cpu ' .. " on " .. tostring(io.popen('hostname -s'):read()))
      torch.setdefaulttensortype('torch.FloatTensor')
   end

   mapWordIdx2Vector = torch.Tensor()
   mapWordStr2WordIdx = {}
   mapWordIdx2WordStr = {}
   trainDataSet = {}
   validDataSet = {}
   testDataSet = {}
   trainDataTensor = torch.Tensor()
   trainDataTensor_y = torch.Tensor()
   trainDataTensor_y_2 = {}
   validDataTensor = torch.Tensor()
   validDataTensor_y = {}
   testDataTensor = torch.Tensor()
   testDataTensor_y = {}

   -- pclient   
   pc = pClient(conf)
   -- go
   if opt.preloadBinary == false then
      dofile('prepareData.lua')
      if oncuda and role == 'pt' then
         trainDataTensor =  trainDataTensor:cuda()
         trainDataTensor_y =  trainDataTensor_y:cuda()
         validDataTensor = validDataTensor:cuda()
         testDataTensor = testDataTensor:cuda()
         torch.setdefaulttensortype('torch.CudaTensor')
      end

   else
      mapWordIdx2Vector = torch.load("binary_mapWordIdx2Vector")
      mapWordStr2WordIdx = torch.load("binary_mapWordStr2WordIdx")
      mapWordIdx2WordStr = torch.load("binary_mapWordIdx2WordStr")
      mapLabel2AnswerIdx = torch.load("binary_mapLabel2AnswerIdx")
      trainDataSet = torch.load("binary_trainDataSet")
      validDataSet = torch.load("binary_validDataSet")
      testDataSet1 = torch.load("binary_testDataSet1")
      testDataSet2 = torch.load("binary_testDataSet2")
   end
   dofile('cnn.lua')
end

mpiT.Finalize()
