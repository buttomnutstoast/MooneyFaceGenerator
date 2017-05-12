local NET = {}
function NET.packages()
    require 'cudnn'
    require 'dpnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local nn4 = torch.load(opt.nn4model)
    local cnn = nn4:get(1)
    local classifier = nn4:get(2)
    local model = nn.Sequential()
        :add(nn.Squeeze(1))
        :add(cnn)
        :add(nn.ConcatTable()
            :add(classifier)
            :add(nn.Identity()))

    return model
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.ClassNLLCriterion())
    return criterion
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('prediction',0,0, true)
    info[#info+1] = utilfuncs.newInfoEntry('inds',0,0, true)
    info[#info+1] = utilfuncs.newInfoEntry('features',0,0, true)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local maxVals, maxInds = outputs[1][{{},{1}}]:max(1)
    local ind = maxInds:squeeze()
    local outputsCPU = outputs[1][{{ind}, {}}]:float()

    info[1].value = outputsCPU
    info[2].value = maxInds
    info[3].value = outputs[2][{{ind}, {}}]:float()
    info[4].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[4].N     = 1
end

function NET.ftest(inputs, labels, model, criterion)
    outputs = model:forward(inputs)
    return outputs, 0
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
