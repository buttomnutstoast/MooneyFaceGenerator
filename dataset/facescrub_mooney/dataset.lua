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
      A dataset class for images in FaceScrub
    ]],
    {name="path",
     type="string",
     help="path to root dir"},

    {name="protocol",
     type="string",
     help="train | test",
     default="train"},

}

function dataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    ----------------------------------------------------------------------
    -- training/testing/eval partition list file
    local pathsFace = {
        'FaceScrub/img_list.txt'
    }

    local tbFace = {}
    for _, path in ipairs(pathsFace) do
        local path_ = paths.concat(self.path, path)
        local dir_ = paths.dirname(path_)
        local tb_ = readTxt(path_, dir_)
        for _, imgPath_ in ipairs(tb_) do
            tbFace[#tbFace+1] = imgPath_
        end
    end

    local imagePath, imageClass = {}, {}
    for i=1,#tbFace do
        imagePath[#imagePath+1] = tbFace[i]
        imageClass[#imageClass+1] = 1
    end

    -- get imagePath and imageClass
    self.imagePath = tb2Tensor(imagePath)
    self.imageClass = tb2Tensor(imageClass)

    self.classListSample = {}
    self.classListSample[1] = torch.range(1, #tbFace)

    self.classes = {'face'}
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
        local out = self:sampleHookTest(imgpath)
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
