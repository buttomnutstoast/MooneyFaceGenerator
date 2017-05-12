require 'image'
local tf = require 'utils/transforms'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(OPT.cache, 'trainCache.t7')
local testCache = paths.concat(OPT.cache, 'testCache.t7')

-- Check for existence of OPT.data
assert(paths.dirp(OPT.data), 'given data directory not exist!')

------------------------------------------------------------------------
--[[
   Section 0: Create image loader functions each for training ane testing,
   the function in training upscale the shorter side to loadSize, however,
   the one in testing upscale the longer side.
--]]
local resizeImg = tf.Scale(OPT.imageCrop+20)
local randCropImg = tf.RandomCrop(OPT.imageCrop)
local centerCropImg = tf.CenterCrop(OPT.imageCrop)
local randAffine = tf.Affine(15)

local function rmBorder(nPixel)
    return function(input)
        local h, w = input:size(2), input:size(3)
        return input:narrow(2, nPixel, h-2*nPixel):narrow(3, nPixel, w-2*nPixel)
    end
end

local function highpass(size)
    local kernel = torch.DoubleTensor(size, size):fill(1/(size*size))
    return function (input)
        local out = image.convolve(input, kernel)
        return resizeImg(out)
    end
end

local function randhighpass(sizes)
    return function (input)
        local size = sizes[torch.random(1,#sizes)]
        return highpass(size)(input)
    end
end

local smooth = highpass(4)
local randSmooth = randhighpass({2,3,4,5,6})

local function reshape(img)
    if img:dim() == 2 then
        img = img:view(1, img:size(1), img:size(2))
    end
    return img:expand(3,img:size(2), img:size(3))
end

local function adjust(img)
    local meanPixel = img:mean()
    local lB, hB = 0.4, 0.6
    local ratio = 1
    if meanPixel > hB then
        ratio = hB / meanPixel
    elseif meanPixel < lB then
        ratio = lB / meanPixel
    end
    img:mul(ratio)
    img:cmin(1)
    return img
end

local function threshold(img)
    local threshold = torch.uniform(0.45, 0.55)
    img = torch.gt(img, threshold):typeAs(img)
    return img
end

local function mooneyFunc(img)
    local func = torch.uniform() >= 0.5 and image.dilate or image.erode
    local k = torch.random(1, 3)
    if img:dim() == 3 then img = img:squeeze(1) end
    return func(img, torch.ones(k,k):typeAs(img))
end

local function preprocess(img, label, protocol)
    local aug
    if label == 1 then
        aug = {}
        aug[#aug+1] = resizeImg
        aug[#aug+1] = rmBorder(10)
        if protocol == 'train' then aug[#aug+1] = randAffine end
        aug[#aug+1] = threshold
        if protocol == 'train' then aug[#aug+1] = mooneyFunc end
        aug[#aug+1] = reshape
    else
        local smooth_ = protocol == 'train' and randSmooth or smooth
        aug = {
            resizeImg,
            rmBorder(10),
            randAffine,
            smooth_,
            adjust,
            threshold,
            reshape,
        }
    end
    return tf.Compose(aug)(img)
end

local function loadImage(path, label, protocol)
    local img = image.load(path, 1, 'float')
    if img:dim() == 2 then
        img = img:view(1, img:size(1), img:size(2))
    end
    return preprocess(img, label, protocol)
end

local split = 90
------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path, label)
    collectgarbage()
    local input = loadImage(path, label, 'train')
    local out = randCropImg(input)

    return out
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
else
    print('Creating training metadata')
    trainLoader = dataLoader{
        path = paths.concat(OPT.data),
        protocol = 'train',
        split = split,
        }
    torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
collectgarbage()

-- End of train loader section
-----------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
--]]

-- function to load the image
local testHook = function(self, path, label)
    local input = loadImage(path, label, 'test')
    local out = centerCropImg(input)

    return out
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
else
    print('Creating test metadata')
    testLoader = dataLoader{
        path = paths.concat(OPT.data),
        protocol = 'test',
        split = split,
        }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()

-- End of test loader section
---------------------------------------------------------------------
