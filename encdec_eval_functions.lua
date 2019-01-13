--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/21/17
-- Time: 10:02 AM
-- To change this template use File | Settings | File Templates.
--

cjson = require 'cjson'

function sample(encoder, decoder, enc_state, sequence, max_samples, probs)
    local lsm = nn.LogSoftMax():cuda()
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = encoder:forward(sequence)
        sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
    end
    if probs == nil then
        probs = torch.zeros(sequence:size(1), 1):cuda()
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, enc_state:size(3))
    local addition = torch.zeros(sequence:size(1)):cuda()
    probs = torch.cat(probs, addition, 2)
    local output = torch.cat(sequence, addition , 2)
    local dec_h0 = enc_state[{{}, enc_state:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
    local y = decoder:forward({cb:clone(), dec_h0, sequence})
    y = lsm:forward(y)
    local sampled = sampler:forward(y)
    local sample_probs = sampler.prob_values
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    for i=1, probs:size(1) do probs[i][probs:size(2)] = sample_probs[probs:size(2) - 1] end
    if max_samples == 1 or helper.n_to_w[output[1][output:size(2)]] == '</S>' then
        return output, probs
    else
        return sample(encoder, decoder, enc_state, output, max_samples - 1, probs)
    end
end

function attn_sample(encoder, decoder, ing_matrix, enc_state, sequence, max_samples, probs, max_out_len_from_dec)
    if max_out_len_from_dec == nil then
        max_out_len_from_dec = 300
    end
    local lsm = nn.LogSoftMax():cuda()
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = encoder:forward(sequence)
        sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1):repeatTensor(4, max_out_len_from_dec)
    end
    if probs == nil then
        probs = torch.zeros(sequence:size(1), sequence:size(2)):cuda()
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), sequence:size(1), enc_state:size(3))
    local dec_h0 = enc_state[{{}, enc_state:size(2), {}}]  -- grab the last hidden state from the encoder, which will be at index max_in_len
    dec_h0 = dec_h0:repeatTensor(sequence:size(1), 1)
    local y = decoder:forward({cb:clone(), dec_h0, ing_matrix, sequence})
    y = lsm:forward(y)
    local sampled = sampler:forward(y)
    sampled = sampled:reshape(sequence:size(1), sampled:size(1) / sequence:size(1))
    local sample_probs = sampler.prob_values
    sample_probs = sample_probs:reshape(sequence:size(1), sample_probs:size(1) / sequence:size(1))
    local output = sequence:cuda():sub(1, sequence:size(1), 1, 1):cat(sampled:sub(1, sequence:size(1), 1, 299):cuda())
    local probs = probs:cuda():sub(1, sequence:size(1), 1, 1):cat(sample_probs:sub(1, sequence:size(1), 1, 299):cuda())
    if max_samples == 1 or helper.n_to_w[output[1][output:size(2)]] == '<end>' then
        return output, probs
    else
        return attn_sample(encoder, decoder, ing_matrix, enc_state, output, max_samples - 1, probs, max_out_len_from_dec)
    end
end

function lm_sample(decoder, enc_state, sequence, max_samples, probs)
    local lsm = nn.LogSoftMax():cuda()
    if max_samples == nil then
        max_samples = 1
    end
    if enc_state == nil then
        enc_state = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, 1, decoder._rnns[1].hidden_dim)
        if sequence == nil then
            sequence = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
        end
    end
    if probs == nil then
        probs = torch.zeros(sequence:size(1), 1):cuda()
    end
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), 1, enc_state:size(3))
    local addition = torch.zeros(sequence:size(1)):cuda()
    probs = torch.cat(probs, addition, 2)
    local output = torch.cat(sequence, addition , 2)
    local dec_h0 = enc_state[{{}, enc_state:size(2), {}}] -- grab the last hidden state
    print(torch.sum(dec_h0))
    print(sequence:size())
    local y = decoder:forward({cb:clone(), dec_h0, sequence})
    y = lsm:forward(y)
    local sampled = sampler:forward(y)
    local sample_probs = sampler.prob_values
    for i=1, output:size(1) do output[i][output:size(2)] = sampled[output:size(2) - 1] end
    for i=1, probs:size(1) do probs[i][probs:size(2)] = sample_probs[probs:size(2) - 1] end
    if max_samples == 1 or helper.n_to_w[output[1][output:size(2)]] == '</S>' then
        return output, probs
    else
        return lm_sample(decoder, enc_state, output, max_samples - 1, probs)
    end
end

function sequence_to_string(seq)
    local str = ''
    if seq:dim() == 2 then seq = seq[1] end
    for i=1, seq:size()[1] do
        local next_word = helper.n_to_w[seq[i]] == nil and '<UNK2>' or helper.n_to_w[seq[i]]
        str = str..' '..next_word
    end
    return str
end

function generate_samples(data_set, outputs, num_samples, max_sample_length, gen_inputs, ingredients)
    if max_sample_length == nil then max_sample_length = 10 end
    local results = {}
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % data_set:size(1)) + 1
        if gen_inputs ~= nil then
            t_set_idx = (gen_inputs[i] % data_set:size(1)) + 1
        end
        if t_set_idx > data_set:size(1) then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if gen_inputs ~= nil then
            example_no = (gen_inputs[i] % example:size(1)) + 1
        end
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        x = x:reshape(1, x:size(1))
        local result = {}
        result['encoder_input'] = sequence_to_string(x)
        local sample_seq, sample_probs = sample(enc, dec, nil, x, max_sample_length)
        print(sequence_to_string(sample_seq))
        print(sample_probs)
        result['generated'] = sequence_to_string(sample_seq)
        result['probabilities'] = sample_probs:totable()
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        if ingredients ~= nil then
            local count = 0
            for j=1, sample_seq:size(1) do
                if ingredients[i][sample_seq[j]] ~= nil then
                    count = count + 1
                end
            end
            result['ingredients_matched'] = count
            result['total_ingredients'] = ingredients[i]['total']
        end
        results[#results + 1] = result
    end
    return results
end

function attn_generate_samples(data_set, outputs, num_samples, max_sample_length, gen_inputs, ingredients, beam_decode)
    -- table which allows us to pull out ingredients form input via tensor:index
    if beam_decode then
        require 'beam_decode'
    end
    local ingredients_index = {}
    for i=2, data_set:size(3) do
        ingredients_index[#ingredients_index + 1] = i
    end
    ingredients_index = torch.CudaLongTensor(ingredients_index)

    if max_sample_length == nil then max_sample_length = 10 end
    local results = {}
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % data_set:size(1)) + 1
        if gen_inputs ~= nil then
            t_set_idx = (gen_inputs[i] % data_set:size(1)) + 1
        end
        if t_set_idx > data_set:size(1) then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if gen_inputs ~= nil then
            example_no = (gen_inputs[i] % example:size(1)) + 1
        end
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        x = x:reshape(1, x:size(1), x:size(2))
        local result = {}
        result['encoder_input'] = sequence_to_string(x[1][1])
        result['ingredients'] = {}
        for i=2, x[1]:size(1) do
            result['ingredients'][#result['ingredients'] + 1] = sequence_to_string(x[1][i])
        end
        local enc_titles = x:select(2, 1)
        print('Generating for: ', sequence_to_string(enc_titles))
        local enc_ingredients = x:index(2, ingredients_index)
        enc_ingredients = enc_ingredients:repeatTensor(4, 1, 1)-- 4 x N x T
        local ing_matrix = make_matrix:forward(enc_ingredients)
        local sample_seq, sample_probs
        if not beam_decode then
            sample_seq, sample_probs = attn_sample(enc, dec, ing_matrix, nil, enc_titles, max_sample_length)
        else
            local dec_h0 = enc:forward(enc_titles)
            print(dec_h0:size())
            local dec_c0 = torch.CudaTensor.zeros(dec_h0:size())
            print(dec_c0:size())
            local beg = torch.CudaTensor({helper.w_to_n['<beg>']}):reshape(1, 1)
            print(beg:size())
            local best = attn_beam_decode_soft(dec, dec._rnns[1], {dec_c0, dec_h0, ing_matrix, beg}, helper.w_to_n['<end>'], 5, helper, false, max_sample_length)
            sample_seq, sample_probs = best['words'], best['p']
        end
        result['generated'] = sequence_to_string(sample_seq)
        result['probabilities'] = sample_probs:totable()
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        if ingredients ~= nil then
            local count = 0
            for j=1, sample_seq:size(1) do
                if ingredients[i][sample_seq[j]] ~= nil then
                    count = count + 1
                end
            end
            result['ingredients_matched'] = count
            result['total_ingredients'] = ingredients[i]['total']
        end
        results[#results + 1] = result
    end
    return results
end


function lm_generate_samples(data_set, dec_inputs, outputs, num_samples, max_sample_length, gen_inputs, seq_init_length)
    if max_sample_length == nil then max_sample_length = 10 end
    local results = {}
    print('Generating Samples...')
    for i = 1, num_samples do
        print('Sample ', i)
        local t_set_idx = (torch.random() % data_set:size(1)) + 1
        if gen_inputs ~= nil then
            t_set_idx = (gen_inputs[i] % data_set:size(1)) + 1
        end
        if t_set_idx > data_set:size(1) then t_set_idx = 1 end
        local example = data_set[t_set_idx]
        local decoding = dec_inputs[t_set_idx]
        local example_no = torch.random() % example:size(1) + 1
        if gen_inputs ~= nil then
            example_no = (gen_inputs[i] % example:size(1)) + 1
        end
        if example_no > example:size(1) then example_no = 1 end
        local x = example[example_no]
        local y = decoding[example_no]
        x = x:reshape(1, x:size(1))
        y = y:sub(1, seq_init_length)
        y = y:reshape(1, y:size(1))
        local result = {}
        result['encoder_input'] = sequence_to_string(x)
        local sample_seq, sample_probs = lm_sample(dec, nil, y, max_sample_length)
        print(sequence_to_string(sample_seq))
        print(sample_probs)
        result['generated'] = sequence_to_string(sample_seq)
        result['probabilities'] = sample_probs:totable()
        result['gold'] = sequence_to_string(outputs[t_set_idx][example_no])
        results[#results + 1] = result
    end
    return results
end

-- calculate perplexity
function perplexity_over_dataset(enc, dec, enc_inputs, dec_inputs, in_lengths, out_lengths, outputs)
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), enc_inputs[1]:size(1), enc:forward(enc_inputs[1]):size(3))
    local data_perplexity = 0
    local data_loss = 0
    for i=1,enc_inputs:size(1) do
        for _,v in pairs(enc._rnns) do v:resetStates() end
        for _,v in pairs(dec._rnns) do v:resetStates() end
        local enc_input = enc_inputs[i]
        local dec_input = dec_inputs[i]
        local output = outputs[i]
        local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
        local dec_h0 = enc_fwd[{{}, enc_inputs[1]:size(2), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
        local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, enc_input:size(1), dec_input:size(2), #helper.n_to_w)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (torch.sum(out_lengths[i]) / enc_inputs[i]:size(1))
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / enc_inputs:size(1))
        data_loss = data_loss + (loss / enc_inputs:size(1))
    end
    return data_perplexity, data_loss
end

function attn_perplexity_over_dataset(enc, dec, make_matrix, enc_inputs, dec_inputs, in_lengths, out_lengths, outputs)
    local data_perplexity = 0
    local data_loss = 0
    -- table which allows us to pull out ingredients form input via tensor:index
    local ingredients_index = {}
    for i=2, enc_inputs:size(3) do
        ingredients_index[#ingredients_index + 1] = i
    end
    ingredients_index = torch.CudaLongTensor(ingredients_index)
    for i=1,enc_inputs:size(1) do
        for _, v in pairs(enc._rnns) do  v:resetStates() end
        for _, v in pairs(dec._rnns) do  v:resetStates() end

        -- retrieve inputs for this batch
        local enc_input = enc_inputs[i] -- 4 X T
        local enc_titles = enc_input:select(2, 1) -- 4 x (N + 1) x T ==> 4 x T
        local enc_ingredients = enc_input:index(2, ingredients_index) -- 4 x N x T
        local dec_input = dec_inputs[i]:squeeze() -- 4 X 300
        local output = outputs[i]:squeeze()
        local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), enc_inputs[1]:size(1), enc:forward(enc_titles):size(3))

        -- forward pass
        local enc_fwd = enc:forward(enc_titles) -- enc_fwd is h1...hN
        local dec_h0 = enc_fwd[{{}, enc_inputs[1]:size(3), {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
        local ing_matrix = make_matrix:forward(enc_ingredients)
        local dec_fwd = dec:forward({cb:clone(), dec_h0, ing_matrix, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, enc_input:size(1), dec_input:size(2), #helper.n_to_w)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (torch.sum(out_lengths[i]) / enc_inputs[i]:size(1))
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / enc_inputs:size(1))
        data_loss = data_loss + (loss / enc_inputs:size(1))
    end
    return data_perplexity, data_loss
end

function lm_perplexity_over_dataset(dec, dec_inputs, out_lengths, outputs, hidden_size)
    local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), dec_inputs[1]:size(1), hidden_size)
    local data_perplexity = 0
    local data_loss = 0
    for i=1,dec_inputs:size(1) do
        for _,v in pairs(dec._rnns) do v:resetStates() end
        local dec_input = dec_inputs[i]
        local output = outputs[i]
        local dec_fwd = dec:forward({cb:clone(), cb:clone(), dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
        dec_fwd = torch.reshape(dec_fwd, dec_input:size(1), dec_input:size(2), #helper.n_to_w)
        local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
        loss = loss / (torch.sum(out_lengths[i]) / dec_inputs[i]:size(1))
        local batch_perplexity = torch.exp(loss)
        data_perplexity = data_perplexity + (batch_perplexity / dec_inputs:size(1))
        data_loss = data_loss + (loss / dec_inputs:size(1))
    end
    return data_perplexity, data_loss
end

function cmdout (cmd)
   local f = io.popen(cmd)
   local lout = f:read("*a")
   f:close()
   return lout
end

function calculate_bleu(references, candidates)
   if reference == nil then reference = 'reference.txt' end
   if candidate == nil then candidate = 'candidate.txt' end
    local io = require 'io'
    local f = io.open(reference, 'w+')
    for i=1, #references do
        f:write(references[i] .. '\n')
    end
    f:close()
    local f = io.open(candidate, 'w+')
    for i=1, #candidates do
        f:write(candidates[i] .. '\n')
    end
    f:close()
    local cmd = 'perl multi-bleu.perl ' .. reference ..' < ' .. candidate .. ' | python parse_bleu.py'
    local s = cmdout(cmd)
    return cjson.decode(s)
end

function n_pairs_bleu(generations, n)
    local refs = {}
    local cands = {}
    for i=1, n do
        local i = (torch.random() % #generations) + 1
        local j = (torch.random() % #generations) + 1
        while i == j and #generations > 1 do
            j = (torch.random() % #generations) + 1
        end
        refs[#refs + 1] = generations[i]:gsub('\'', '\\\'')
        cands[#cands + 1] = generations[j]:gsub('\'', '\\\'')

    end
    return calculate_bleu(refs, cands)
end

function closest_bleu_match(references, samples, ref_file, sample_file)
    if ref_file == nil then ref_file = 'ref_file.txt' end
    if sample_file == nil then sample_file = 'sample_file.txt' end
    local io = require 'io'
    local f = io.open(ref_file, 'w+')
    f:write(cjson.encode(references))
    f:close()
    local f = io.open(sample_file, 'w+')
    f:write(cjson.encode(samples))
    f:close()
    local cmd = 'python closest_bleu_match.py ' .. sample_file ..' ' .. ref_file
    local s = cmdout(cmd)
    return cjson.decode(s)
end

function alignment_scores(sequences)
    local cmd = 'python alignment.py ' .. '\'' .. cjson.encode(sequences):gsub('\'', '') .. '\''
    local s = cmdout(cmd)
    local t = cjson.decode(s)
    local data = torch.DoubleTensor(t)
    local avg_alignment_score = 0; local count = 0
    for i=1, #sequences - 1 do for j=i + 1, #sequences do
        avg_alignment_score = avg_alignment_score + data[i][j]; count = count + 1
    end end
    avg_alignment_score = avg_alignment_score / count
    return avg_alignment_score
end
