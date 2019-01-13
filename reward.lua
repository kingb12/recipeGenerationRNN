function reward_maker(target, exclude_num) 
  local function r(sample) 
    local max_subseq=0
    local max_end=0
    local ti = 1
    local lcs = {}
    for i = 0, target:size(1) do
      lcs[i] = {}
      for j = 0, sample:size(1) do
        lcs[i][j] = 0
      end
    end 
    for i = 1, target:size(1) do
      for j = 1, sample:size(1) do
        if target[i] == sample[j] and (exclude_num == nil or target[i] ~= exclude_num) then --so we can ignore characters like <mask>
          lcs[i][j] = lcs[i-1][j-1] + 1
          if lcs[i][j] > max_subseq then
              max_subseq = lcs[i][j]
              max_end = i
          end
        else
          lcs[i][j] = 0
        end
      end
    end
    return max_subseq, (max_end ~= 0 and target:sub(max_end + 1 - max_subseq, max_end) or nil)

  end
  return r
end
