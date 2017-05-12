require 'image'
local tf = require 'utils/transforms'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local testCache = paths.concat(OPT.cache, 'testCache.t7')

-- make sure batchSize = 1
assert(OPT.batchSize == 1, 'please make sure batchSize = 1')
-- Check for existence of OPT.data
assert(paths.dirp(OPT.data), 'given data directory not exist!')

------------------------------------------------------------------------
--[[
   Section 0: Create image loader functions each for training ane testing,
   the function in training upscale the shorter side to loadSize, however,
   the one in testing upscale the longer side.
--]]
local resizeImg = tf.Scale(OPT.imageCrop)
local centerCropImg = tf.CenterCrop(OPT.imageCrop)

local function reshape(img)
    return img:expand(img:size(1), 3, img:size(3), img:size(4))
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

local function highpass(size)
    local kernel = torch.DoubleTensor(size, size):fill(1/(size*size))
    return function (input)
        local out = image.convolve(input, kernel)
        return resizeImg(out)
    end
end

local function jitter(img)
    local sizes = {2,3,4,5,6}
    local ths = {0.4,0.45, 0.475, 0.5, 0.525, 0.55,0.6}
    local outputs = {}
    for _, size in ipairs(sizes) do
        local smooth = highpass(size)
        for _, th in ipairs(ths) do
            local img_ = smooth(img)
            img_ = adjust(img_)
            img_ = img_:ge(th):typeAs(img_)
            outputs[#outputs+1] = img_:view(1, 1, OPT.imageCrop, OPT.imageCrop)
        end
    end
    return torch.cat(outputs, 1)
end


local function preprocess(img)
    local aug = {
        resizeImg,
        centerCropImg,
        jitter,
        reshape,
    }
    return tf.Compose(aug)(img)
end

local function loadImage(path)
    local img = image.load(path, 1, 'float')
    if img:dim() == 2 then
        img = img:view(1, img:size(1), img:size(2))
    end
    return preprocess(img)
end
-----------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
--]]

-- function to load the image
local testHook = function(self, path)
    return loadImage(path)
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
else
    print('Creating test metadata')
    testLoader = dataLoader{
        path = paths.concat(OPT.data),
        protocol = 'test',
        }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()

-- End of test loader section
---------------------------------------------------------------------
