require 'image'
require 'hdf5'
ffi = require 'ffi'

local tf = dofile('utils/transforms.lua')

-- command options
local cmd = torch.CmdLine()
cmd:option('-hdf5')
local opt = cmd:parse(arg or {})

local resizeImg = tf.Scale(96)
local centerCrop = tf.CenterCrop(96)

local function reshape(img)
    if img:dim() == 2 then
        img:resize(1, img:size(1), img:size(2))
    end
    return img
end

local function threshold(th)
    return function (input)
        return input:gt(th):typeAs(input)
    end
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

local function jitters()
    local sizes = {2,3,4,5,6}
    local ths = {0.4, 0.45, 0.475, 0.5, 0.525, 0.55, 0.6}
    local funcs = {}
    for _, size in ipairs(sizes) do
        local smooth = highpass(size)
        for _, th in ipairs(ths) do
            local funcs_ = {
                smooth,
                adjust,
                threshold(th)
            }
            funcs[#funcs+1] = tf.Compose(funcs_)
        end
    end
    return funcs
end

local JITTERS = jitters()
function preprocess(path, ind)
    -- helper function
    local helper = tf.Compose({
        resizeImg,
        centerCrop,
        JITTERS[ind],
        reshape
    })

    local img = image.load(path, 1, 'float')
    if img:dim()==2 then img:resize(1, img:size(1), img:size(2)) end
    return helper(img)
end

function loadImage(path)
    -- helper function
    local helper = tf.Compose({
        resizeImg,
        centerCrop,
    })

    local img = image.load(path, 1, 'float')
    if img:dim()==2 then img:resize(1, img:size(1), img:size(2)) end
    return helper(img)
end

-- load cache
torch.class('dataLoader')
local testCache = torch.load('checkpoint/facescrub_mooney/testCache.t7')
imagePath = testCache.imagePath
-- load result
local h5db = hdf5.open(opt.hdf5, 'r')
prediction = h5db:read('prediction'):all()
indices = h5db:read('inds'):all()
indices = indices:squeeze(2)


local mnyPred = prediction[{{},{1}}]:squeeze(2)
local maxVals, maxInds = mnyPred:sort(1, true)
maxInds = maxInds:squeeze()

-- get statistic
local classInds = {}
local classes = {}
for i = 1,maxVals:size(1) do
    if torch.exp(maxVals[i]) > 0.997 then
        local inds = maxInds[i]
        local path_ = ffi.string(torch.data(imagePath[inds]))
        local tmp = path_:split('/')
        local identity = tmp[#tmp-1]
        classInds[identity] = classInds[identity] or {}
        table.insert(classInds[identity], inds)
    else
        break
    end
end
for class, _ in pairs(classInds) do table.insert(classes, class) end

local splitTb = {
    train = {},
    test = {},
    val = {}
}
local tmp = {
    train = {},
    test = {},
    val = {}
}
local reorder = torch.randperm(#classes)

for i = 1,#classes do
    local split
    if i <= math.ceil(#classes / 3) then
        split = 'train'
    elseif i <= math.ceil(#classes * 2 / 3) then
        split = 'test'
    else
        split = 'val'
    end

    local indices = classInds[classes[reorder[i]]]
    for _, ind in ipairs(indices) do
        table.insert(splitTb[split], ind)
    end
    table.insert(tmp[split], classes[reorder[i]])

    local path_ = ffi.string(torch.data(imagePath[indices[1]]))
end
print(#splitTb['train'], #splitTb['test'], #splitTb['val'])
print(#tmp['train'], #tmp['test'], #tmp['val'])

for _, train_class in ipairs(tmp['train']) do
    for _, val_class in ipairs(tmp['val']) do
        assert(train_class ~= val_class)
    end
end

-- save to disk
local rootDir = 'figures/facescrub'
local bwDir = paths.concat(rootDir, 'bw')
local grayDir = paths.concat(rootDir, 'gray')
if not paths.dirp(rootDir) then paths.mkdir(rootDir) end
if not paths.dirp(bwDir) then paths.mkdir(bwDir) end
if not paths.dirp(grayDir) then paths.mkdir(grayDir) end


for split, inds in pairs(splitTb) do
    for _, ind_ in ipairs(inds) do
        local path_ = ffi.string(torch.data(imagePath[ind_]))
        local tmp = path_:split('/')
        local imgname_ = tmp[#tmp]:gsub('.png', ''):gsub('.jpg', '')
        local img_ = preprocess(path_, indices[ind_])

        local format = string.format('%s.png', imgname_)
        local bwImgName = paths.concat(bwDir, split, format)
        local grayImgName = paths.concat(grayDir, split, format)

        if not paths.dirp(paths.dirname(bwImgName)) then paths.mkdir(paths.dirname(bwImgName)) end
        if not paths.dirp(paths.dirname(grayImgName)) then paths.mkdir(paths.dirname(grayImgName)) end
        image.save(bwImgName, img_)
        image.save(grayImgName, loadImage(path_))
    end
end
