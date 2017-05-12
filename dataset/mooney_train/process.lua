function readTxt(txtpath, imgDir)
    local f = io.open(txtpath, 'r')
    local str = f:read("*all")
    local tb = str:split('\n')
    if imgDir then
        for i = 1,#tb do
            tb[i] = paths.concat(imgDir, tb[i])
        end
    end
    return tb
end

local function maxString(tb)
    local maxLen = 0
    for _, str in ipairs(tb) do
        if #str > maxLen then maxLen = #str end
    end
    return maxLen
end

function tb2Tensor(tb)
    assert(#tb > 0, 'should not input zero-length table')
    local tensor
    if torch.type(tb[1]) == 'string' then
        local ffi = require 'ffi'
        local maxLen = maxString(tb)
        maxLen = maxLen + 1 -- c-style char array
        tensor = torch.CharTensor(#tb, maxLen)
        local s_data = tensor:data()
        for ind, str in ipairs(tb) do
            ffi.copy(s_data, str)
            s_data = s_data + maxLen
        end
    elseif torch.type(tb[1]) == 'number' then
        tensor = torch.FloatTensor(tb)
    end
    return tensor
end
