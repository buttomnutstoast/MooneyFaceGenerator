mathfuncs = {}

function mathfuncs.topK(prediction, target, K)
    local acc = 0
    local _,prediction_sorted = prediction:sort(2, true) -- descending
    local batch_Size = prediction:size(1)
    for i=1,batch_Size do
        for j=1,K do
            if prediction_sorted[i][j] == target[i] then
                acc = acc + 1
                break
            end
        end
    end
    return acc/batch_Size
end

return mathfuncs