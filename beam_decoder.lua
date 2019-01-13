--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 4/17/17
-- Time: 10:36 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'cutorch'
require 'io'
priority_queue = require 'priority_queue'

function beam_decode (net, rnn, beg, fin, width, helper, verbose, max)
    -- net is whole net, rnn is just LSTM, beg is start token, fin is end token
    -- max = max number of generated tokens
    -- width = beam size2

    -- local length = input:size(1)
    local lsm = nn.Sequential()
    lsm:add(nn.View(-1))
    lsm:add(nn.LogSoftMax())
    lsm:cuda()

    local candidates = {
        {
            input=beg,
            words={},
            p=0
        }
    }

    local best_ws, best_p = nil, nil
    for t = 0, max do
        assert(#candidates == width or t == 0)

        if best_ws ~= nil and best_p > candidates[1]['p'] then break end

        local next_candidates = {}
        for i = 1, width do table.insert(next_candidates, {p=-math.huge}) end

        for _, c in pairs(candidates) do
            assert(c['fin'] == nil)
            print(c['input'])
            local out = net:forward(c['input'])
            local hidden, cell = torch.reshape(rnn.output, 1, rnn.output:size(3)), torch.reshape(rnn.cell, 1, rnn.cell:size(3))
            local probs = lsm:forward(out)

            local top = {}
            for i=1, width-#top do table.insert(top, {p=-math.huge}) end

            for i = 1, probs:size(1) do
                local p = probs[i] -- this is where we get most likely

                if p > top[#top]['p'] then
                    local j = #top-1
                    while j > 0 and p > top[j]['p'] do j = j - 1 end --scan to correct insertion locaton
                    j = j + 1
                    for k = #top-1, j, -1 do top[k+1] = top[k] end -- starting from the end -1, shift each to the right
                    top[j] = {w=i, p=p} --insert the new probability in its right place. The word == i because |probs| == vocab_size
                end
                for a = 1, #top-1 do assert(top[a]['p'] >= top[a+1]['p']) end -- sanity check that we still have a sorted order
            end

            for _,u in ipairs(top) do
                local p, w = c['p'] + u['p'], u['w']
                local nc = {}
                local word = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, 1)
                word[1][1] = w
                nc['input'] = {cell, hidden, word}
                local ws = {}
                for i = 1, #c['words'] do table.insert(ws, c['words'][i]) end
                table.insert(ws, w)
                nc['words'] = ws
                nc['p'] = p
                if w == fin then
                    if best_p == nil or nc['p'] > best_p then
                        best_p = nc['p']
                        best_ws = {}
                        for i=1,#ws do table.insert(best_ws, ws[i]) end
                    end
                else
                    if p > next_candidates[#next_candidates]['p'] and (not strong or nc['gt'] >= min_gt) then
                        j = #next_candidates -1
                        while j > 0 and p > next_candidates[j]['p'] do j = j - 1 end
                        j = j + 1
                        local same = true
                        if j ~= 1 and #next_candidates[j-1]['words'] == #nc['words'] then
                            for k = 1, #nc['words'] do same = same and next_candidates[j-1]['words'][k] == nc['words'][k] end
                        else
                            same = false
                        end
                        if not same then
                            for k = #next_candidates -1, j, -1 do next_candidates[k+1] = next_candidates[k] end
                            next_candidates[j] = nc
                        end
                    end
                end
                for a = 1, #next_candidates -1 do assert(next_candidates[a]['p'] >= next_candidates[a+1]['p']) end
            end
        end

        if verbose > 0 then
            print('t = ' .. t)
            for i = 1, #candidates do
                if candidates[i]['words'] ~= nil then
                    io.write('  ')
                    for j = 1, #candidates[i]['words'] do
                        io.write(helper.n_to_w[candidates[i]['words'][j]] .. ' ')
                    end
                end
                print(candidates[i]['p'])
            end
            if verbose > 1 then
                io.read()
            end
        end

        candidates = next_candidates
    end

    if best_ws == nil then
        best_ws, best_p = candidates[1]['words'], candidates[2]['p']
    end
    best = {}
    best['words'] = best_ws
    best['p'] = best_p
    return best
end

function push (net, rnn, beg, output)
    local lsm = nn.Sequential()
    lsm:add(nn.View(-1))
    lsm:add(nn.LogSoftMax())
    lsm:cuda()

    local u = {
        input=beg,
        words={},
        p=0
    }


    net:forget(); net:evaluate()
    for t = 1, output:size(1) do
        assert(#u['words'] == t-1)
        out = net:forward(u['input'])
        hidden, cell = torch.reshape(rnn.output, 1, rnn.output:size(3)), torch.reshape(rnn.cell, 1, rnn.cell:size(3))
        local ps = lsm:forward(out)
        local w = output[t]
        local word = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, 1)
        word[1][1] = w
        u['input'] = {cell, hidden, word}
        table.insert(u['words'], w)
        u['p'] = u['p'] + ps[w]
    end

    return u
end

function beam_decode_soft(net, rnn, beg, fin, width, helper, verbose, max)
    local function cmp_candidates(a, b)
        return a.p < b.p
    end

    -- for this to make sense, only send in one example at a time
    local soft_max = nn.Sequential()
    soft_max:add(nn.View(-1))
    soft_max:add(nn.SoftMax())
    soft_max:cuda()

    local best_ws, best_p = nil, nil

    local candidates = priority_queue.new(cmp_candidates, {
        {
            input=beg,
            words={},
            p=0
        }
    })

    for t = 0, max do
        local next_candidates = priority_queue.new(cmp_candidates)

        -- for each candidate, examine K new candidates
        while #candidates > 0 do
            local c = candidates:pop()
            assert(c['fin'] == nil)
            local out = net:forward(c['input'])
            local hidden, cell = torch.reshape(rnn.output, 1, rnn.output:size(3)), torch.reshape(rnn.cell, 1, rnn.cell:size(3))
            local probs = soft_max:forward(out)
            local next_words  = torch.multinomial(probs, width):cudaLong()  -- k next words for this candidates
            local next_probs = probs:index(1, next_words)

            for i=1, next_words:size(1) do
                -- preparre next candidate from a new one
                local next_c = {}
                local p, w = c['p'] + math.log(next_probs[i]), next_words[i] -- use log probs to prevent underflow
                local word = torch.CudaTensor(1, 1)
                word[1][1] = w
                next_c['input'] = {cell, hidden, word}
                local words = {}
                for i = 1, #c['words'] do table.insert(words, c['words'][i]) end
                table.insert(words, w)
                next_c['words'] = words
                next_c['p'] = p

                -- if we're done, check if its the best
                if w == fin then
                    if best_p == nil or next_c['p'] > best_p then
                        best_p = next_c['p']
                        best_ws = {}
                        for i=1,#words do table.insert(best_ws, words[i]) end
                    end
                else
                    next_candidates:push(next_c)
                end
            end
        end
        -- drop the K^2 - K lowest ones
        while t > 2 and #next_candidates > width do
            next_candidates:pop()
        end
        candidates = next_candidates

        -- show progress if desired
        if verbose > 0 then
            print('t = ' .. t)
            for i = 1, #candidates do
                if candidates[i]['words'] ~= nil then
                    io.write('  ')
                    for j = 1, #candidates[i]['words'] do
                        io.write(helper.n_to_w[candidates[i]['words'][j]] .. ' ')
                    end
                end
                print(candidates[i]['p'])
            end
            if verbose > 1 then
                io.read()
            end
        end
    end

    -- if we are at the end but never hit an <end> token, choose the best available
    if best_ws == nil then
        best_p = -math.huge
        for i=1, #candidates do
            if candidates[i].p > best_p then
                best_p = candidates[i].p
                best_ws = candidates[i].words
            end
        end
    end
    local best = {}
    best['words'] = best_ws
    best['p'] = best_p
    return best
end

function attn_beam_decode_soft(net, rnn, beg, fin, width, helper, verbose, max)
    local function cmp_candidates(a, b)
        return a.p < b.p
    end

    -- for this to make sense, only send in one example at a time
    local soft_max = nn.Sequential()
    soft_max:add(nn.View(-1))
    soft_max:add(nn.SoftMax())
    soft_max:cuda()

    local best_ws, best_p = nil, nil

    local candidates = priority_queue.new(cmp_candidates, {
        {
            input=beg,
            words={},
            p=0
        }
    })

    for t = 0, max do
        local next_candidates = priority_queue.new(cmp_candidates)

        -- for each candidate, examine K new candidates
        while #candidates > 0 do
            local c = candidates:pop()
            assert(c['fin'] == nil)
            local out = net:forward({c['input']})
            local hidden, cell = torch.reshape(rnn.output, 1, rnn.output:size(3)), torch.reshape(rnn.cell, 1, rnn.cell:size(3))
            local probs = soft_max:forward(out)
            local next_words  = torch.multinomial(probs, width):cudaLong()  -- k next words for this candidates
            local next_probs = probs:index(1, next_words)

            for i=1, next_words:size(1) do
                -- preparre next candidate from a new one
                local next_c = {}
                local p, w = c['p'] + math.log(next_probs[i]), next_words[i] -- use log probs to prevent underflow
                local word = torch.CudaTensor(1, 1)
                word[1][1] = w
                next_c['input'] = {cell, hidden,c['input'][3], word}
                local words = {}
                for i = 1, #c['words'] do table.insert(words, c['words'][i]) end
                table.insert(words, w)
                next_c['words'] = words
                next_c['p'] = p

                -- if we're done, check if its the best
                if w == fin then
                    if best_p == nil or next_c['p'] > best_p then
                        best_p = next_c['p']
                        best_ws = {}
                        for i=1,#words do table.insert(best_ws, words[i]) end
                    end
                else
                    next_candidates:push(next_c)
                end
            end
        end
        -- drop the K^2 - K lowest ones
        while t > 2 and #next_candidates > width do
            next_candidates:pop()
        end
        candidates = next_candidates

        -- show progress if desired
        if verbose > 0 then
            print('t = ' .. t)
            for i = 1, #candidates do
                if candidates[i]['words'] ~= nil then
                    io.write('  ')
                    for j = 1, #candidates[i]['words'] do
                        io.write(helper.n_to_w[candidates[i]['words'][j]] .. ' ')
                    end
                end
                print(candidates[i]['p'])
            end
            if verbose > 1 then
                io.read()
            end
        end
    end

    -- if we are at the end but never hit an <end> token, choose the best available
    if best_ws == nil then
        best_p = -math.huge
        for i=1, #candidates do
            if candidates[i].p > best_p then
                best_p = candidates[i].p
                best_ws = candidates[i].words
            end
        end
    end
    local best = {}
    best['words'] = best_ws
    best['p'] = best_p
    return best
end
