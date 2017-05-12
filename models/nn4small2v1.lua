local function createNN4()
   local net = nn.Sequential()

   net:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())

   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))

   -- Inception (2)
   net:add(nn.SpatialConvolutionMM(64, 64, 1, 1))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())
   net:add(nn.SpatialConvolutionMM(64, 192, 3, 3, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(192))
   net:add(nn.ReLU())

   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

   -- Inception (3a)
   net:add(nn.Inception{
     inputSize = 192,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 32},
     reduceSize = {96, 16, 32, 64},
     pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
     batchNorm = true
   })

   -- Inception (3b)
   net:add(nn.Inception{
     inputSize = 256,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 64},
     reduceSize = {96, 32, 64, 64},
     pool = nn.SpatialLPPooling(256, 2, 3, 3, 1, 1),
     batchNorm = true
   })

   -- Inception (3c)
   net:add(nn.Inception{
     inputSize = 320,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 64},
     reduceSize = {128, 32, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
     batchNorm = true
   })

   -- Inception (4a)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {192, 64},
     reduceSize = {96, 32, 128, 256},
     pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
     batchNorm = true
   })

   -- Inception (4e)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 128},
     reduceSize = {160, 64, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
     batchNorm = true
   })

   -- Inception (5a)
   net:add(nn.Inception{
              inputSize = 1024,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {96, 96, 256},
              pool = nn.SpatialLPPooling(960, 2, 3, 3, 1, 1),
              batchNorm = true
   })

   -- Inception (5b)
   net:add(nn.Inception{
              inputSize = 736,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {96, 96, 256},
              pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
              batchNorm = true
   })

   net:add(nn.SpatialAveragePooling(3, 3))

   net:add(nn.View(736))
   net:add(nn.Linear(736, 128))
   net:add(nn.Normalize(2))

   return net
end

local NET = {}
function NET.packages()
    require 'cudnn'
    require 'dpnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local nn4 = createNN4()

    local classifier = nn.Sequential()
    classifier:add(nn.Linear(128, 2))
    classifier:add(nn.LogSoftMax())

    local model = nn.Sequential():add(nn4):add(classifier)

    return model
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.ClassNLLCriterion())
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    -- utilfuncs.newInfoEntry is defined in utils/train_eval_test_func.lua
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()

    info[1].value   = err * iterSize
    info[1].N       = batch_size

    info[2].value   = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('prediction',0,0, true)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value = outputsCPU
    info[2].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N     = batch_size
end

function NET.gradProcessing(model, modelPa, modelGradPa, currentEpoch)
    -- [1-150]: nn4, [151-152]: linear-classifier
    for i = 1,150 do modelGradPa[i]:mul(0.1) end
end

function NET.trainRule(currentEpoch)
    local delta = 2 
    local startVal = 2 -- start from 1e-3 to 1e-(3+delta)
    return {LR= 10^-((currentEpoch-1)*delta/(OPT.nEpochs-1)+startVal),
            WD= 5e-4}
end

return NET
