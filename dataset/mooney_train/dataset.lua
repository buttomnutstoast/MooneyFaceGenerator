require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

paths.dofile('process.lua')

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
      A dataset class for images in Mooney and ImageNet
    ]],
    {name="path",
     type="string",
     help="path to root dir"},

    {name="protocol",
     type="string",
     help="train | test",
     default="train"},

    {name="split",
     type="number",
     help="Percentage of split to go to Training"
    },
}

function dataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    ----------------------------------------------------------------------
    -- training/testing/eval partition list file
    local pathsMny = {
        'Mooney/img_list.txt'
    }
    local pathsNonMny = {
        'Mooney/rev_img_list.txt',
        'Mooney/vflip_img_list.txt',
        'Mooney/vflip_rev_img_list.txt',
        'ILSVRC/sample_img_list.txt',
        'ILSVRC/rev_sample_img_list.txt',
    }

    local tbMny, tbNonMny = {}, {}
    -- get mooney faces
    for _, path in ipairs(pathsMny) do
        local path_ = paths.concat(self.path, path)
        local dir_ = paths.dirname(path_)
        local tb_ = readTxt(path_, dir_)
        for _, imgPath_ in ipairs(tb_) do
            tbMny[#tbMny+1] = imgPath_
        end
    end

    -- get non-mooney faces
    local ind2NonMnyPath = {}
    for _, path in ipairs(pathsNonMny) do
        local path_ = paths.concat(self.path, path)
        local dir_ = paths.dirname(path_)
        local tb_ = readTxt(path_, dir_)
        for _, imgPath_ in ipairs(tb_) do
            tbNonMny[#tbNonMny+1] = imgPath_
        end
    end

    -- split trainint & testing set
    torch.manualSeed(OPT.manualSeed)
    local mnyInds = torch.randperm(#tbMny)
    torch.manualSeed(OPT.manualSeed)
    local nonMnyInds = torch.randperm(#tbNonMny)

    local mnySplit = math.ceil(#tbMny * self.split / 100)
    local nonMnySplit = math.ceil(#tbNonMny * self.split / 100)

    if self.protocol == 'train' then
        mnyInds = mnyInds[{{1,mnySplit}}]
        nonMnyInds = nonMnyInds[{{1,nonMnySplit}}]
    else
        mnyInds = mnyInds[{{mnySplit+1,#tbMny}}]
        nonMnyInds = nonMnyInds[{{nonMnySplit+1,#tbNonMny}}]
    end

    local imagePath, imageClass = {}, {}
    for i=1,mnyInds:size(1) do
        imagePath[#imagePath+1] = tbMny[mnyInds[i]]
        imageClass[#imageClass+1] = 1
    end
    for i=1,nonMnyInds:size(1) do
        imagePath[#imagePath+1] = tbNonMny[nonMnyInds[i]]
        imageClass[#imageClass+1] = 2
    end

    -- get imagePath and imageClass
    self.imagePath = tb2Tensor(imagePath)
    self.imageClass = tb2Tensor(imageClass)

    self.classListSample = {}
    self.classListSample[1] = torch.range(1, mnyInds:size(1))
    self.classListSample[2] = torch.range(
        mnyInds:size(1)+1,
        mnyInds:size(1)+nonMnyInds:size(1)
        )

    self.classes = {'mooney','non-mooney'}
end

function dataset:size()
    return self.imagePath:size(1)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, tab)
    local tensor
    local quantity = #tab
    local iSize = torch.isTensor(tab[1]) and tab[1]:size():totable() or {}
    local tSize = {quantity}
    for _, dim in ipairs(iSize) do table.insert(tSize, dim) end
    tensor = torch.Tensor(table.unpack(tSize)):fill(-1)
    for i=1,quantity do
        tensor[i] = tab[i]
    end
    return tensor
end

function dataset:getByClass(class)
    local index = math.max(1, math.ceil(torch.uniform() * self.classListSample[class]:nElement()))
    local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
    return self:sampleHookTrain(imgpath, class)
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
    assert(quantity)
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        local dice = torch.uniform()
        local class = dice >= 0.5 and 1 or 2
        local out = self:getByClass(class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:genInputs(quantity, currentEpoch)
    local data, scalarLabels = self:sample(quantity)
    return {data}, {scalarLabels}
end

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
        local class = self.imageClass[indices[i]]
        local out = self:sampleHookTest(imgpath, class)
        table.insert(dataTable, out)
        table.insert(scalarTable,  class)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:getInputs(i1, i2, currentEpoch)
    local data, scalarLabels = self:get(i1, i2)
    return {data}, {scalarLabels}
end

return dataset
