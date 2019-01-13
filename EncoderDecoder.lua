--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/8/17
-- Time: 10:53 AM
-- To change this template use File | Settings | File Templates.
--

package.path = ';/homes/iws/kingb12/LanguageModelRNN/?.lua;'..package.path

require 'torch'
require 'nn'
require 'nnx'
require 'nngraph'
require 'util'
require 'math'
require 'learning_rate'
-- require 'torch-rnn'
require 'DynamicView'
require 'Sampler'
require 'optim'
require 'LSTM'
require 'TemporalCrossEntropyCriterion'
require 'encdec_eval_functions'
require 'reward'
cjson = require 'cjson'
io = require 'io'

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-enc_inputs', '../data/rl_enc_inputs.th7')
cmd:option('-dec_inputs', '../data/rl_dec_inputs.th7')
cmd:option('-outputs', '../data/rl_outputs.th7')
cmd:option('-in_lengths', '../data/rl_in_lengths.th7')
cmd:option('-out_lengths', '../data/rl_out_lengths.th7')
cmd:option('-helper', '../data/rl_helper.th7')

cmd:option('-data_config', '', 'load data from a config file. also allows specifying multiple sets')

cmd:option('-valid_enc_inputs', '/homes/iws/kingb12/data/rl_venc_inputs.th7')
cmd:option('-valid_dec_inputs', '/homes/iws/kingb12/data/rl_vdec_inputs.th7')
cmd:option('-valid_outputs', '/homes/iws/kingb12/data/rl_voutputs.th7')
cmd:option('-valid_in_lengths', '/homes/iws/kingb12/data/rl_vin_lengths.th7')
cmd:option('-valid_out_lengths', '/homes/iws/kingb12/data/rl_vout_lengths.th7')

cmd:option('-max_in_len', 200, 'max encoder sequence length')
cmd:option('-max_out_len', 300, 'max decoder sequence length')
cmd:option('-min_out_len', 1, 'min encoder sequence length')
cmd:option('-min_in_len', 1, 'min decoder sequence length')
cmd:option('-batch_size', 4)
cmd:option('-stop_criteria_num_epochs', 0, 'cutoff for number of epochs of increaseing valid loss after which to stop')

-- Model options
cmd:option('-init_enc_from', '')
cmd:option('-init_dec_from', '')
cmd:option('-wordvec_size', 100)
cmd:option('-hidden_size', 512)
cmd:option('-dropout', 0)
cmd:option('-enc_dropout', 0)
cmd:option('-dec_dropout', 0)
cmd:option('-dropout_loc', 'after')
cmd:option('-enc_dropout_loc', 'after')
cmd:option('-dec_dropout_loc', 'after')


cmd:option('-num_enc_layers', 1)
cmd:option('-num_dec_layers', 1)
cmd:option('-weights', '')
cmd:option('-no_average_loss', false)
cmd:option('-enc_forget_states', false)
cmd:option('-dec_forget_states', false)
cmd:option('-bag_of_words', '', 'encoder is replaced with a bag of words approach')
cmd:option('-bow_no_linear', false, 'only meaningful in bag_of_words context. no linear projection to hidden size, all are wordvec_size')

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.0)
cmd:option('-algorithm', 'adam')
cmd:option('-reinforcement', false)
cmd:option('-track_reinforcement_reward_every',0)
cmd:option('-reward_threshold', 1)
cmd:option('-first_reinforce_epoch', 10, "The first epoch in which reinforcement training will be used")
cmd:option('-max_resample_count', 4, "The max number of times in reinforcement in which we resample to try and find a better one")

--Output Options
cmd:option('-print_loss_every', 1000)
cmd:option('-save_model_at_epoch', false)
cmd:option('-save_prefix', '/homes/iws/kingb12/LanguageModelRNN/')
cmd:option('-backup_save_dir', '')
cmd:option('-run', false)
cmd:option('-print_acc_every', 0)
cmd:option('-print_examples_every', 0, 'how often to print out samples')
cmd:option('-valid_loss_every', 0)
cmd:option('-monitor_enc_states', false,  'When set to true, monitors Encoder hiddne state distances')
cmd:option('-monitor_outputs', false,  'When set to true, monitors Encoder hiddne state distances')


-- Backend options
cmd:option('-gpu', false)

local opt = cmd:parse(arg)
local tensorType = 'torch.FloatTensor'
local learningRate = opt.learning_rate
local dropout = opt.dropout > 0.0

-- Choosing between GPU/CPU Mode
if opt.gpu then
    tensorType = 'torch.CudaTensor'
    require 'cutorch'
    require 'cunn'
end

-- ============================================= DATA ==================================================================

--save inintialization
if opt.save_model_at_epoch then
    local f = io.open(opt.save_prefix..'_cmd.json', 'w+')
    f:write(cjson.encode(opt))
    f:close()
end

-- loaded from saved torch files
if opt.data_config ~= '' then
    local f = io.open(opt.data_config)
    data_config = cjson.decode(f:read())
    input_section = 1
    enc_inputs = torch.load(data_config.enc_inputs[1]) -- title <begin> ingredients
    dec_inputs = torch.load(data_config.dec_inputs[1]) -- recipe
    outputs = torch.load(data_config.outputs[1]) -- recipe shifted one over (like a LM)
    in_lengths = torch.load(data_config.in_lengths[1]) -- lengths specifying end of padding
    out_lengths = torch.load(data_config.out_lengths[1]) -- lengths specifying end of padding
    helper = torch.load(data_config.helper) -- has word_map, reverse, etc.
else
    enc_inputs = torch.load(opt.enc_inputs) -- title <begin> ingredients
    dec_inputs = torch.load(opt.dec_inputs) -- recipe
    outputs = torch.load(opt.outputs) -- recipe shifted one over (like a LM)
    in_lengths = torch.load(opt.in_lengths) -- lengths specifying end of padding
    out_lengths = torch.load(opt.out_lengths) -- lengths specifying end of padding
    helper = torch.load(opt.helper) -- has word_map, reverse, etc.
end
local vocab_size = #helper.n_to_w
local wmap = helper.w_to_n
local pad_num = wmap['<pad>']
local beg_num = wmap['<beg>']
local end_num = wmap['<end>']
-- =========================================== THE MODEL ===============================================================

-- ***** ENCODER *****

local lu = nn.LookupTable(vocab_size, opt.wordvec_size)
lu.weight = torch.uniform(lu.weight, -0.1, 0.1)
local enc_lu, dec_lu = lu, lu:clone('weight', 'gradWeight')
if opt.init_enc_from == '' then
    -- The Word Embedding Layer --
    -- word-embeddings can be learned using a LookupTable. Training is faster if they are supplied pre-trained, which can be done by changing
    -- the weights at the index for a given word to its embedding form word2vec, etc. This is a doable next-step
    enc = nn.Sequential()
    -- Input Layer: Embedding LookupTable
    enc:add(enc_lu) -- takes a sequence of word indexes and returns a sequence of word embeddings
    enc._rnns = {}

    -- Hidden Layers: Two LSTM layers, stacked
    -- next steps: dropout, etc.
    for i=1,opt.num_enc_layers do
        local lstm
        if i == 1 then
            lstm = nn.LSTM(opt.wordvec_size, opt.hidden_size)
        else
            lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)
        end
        if not opt.enc_forget_states then
            lstm.remember_states = true
        end
        lstm.weight = torch.uniform(lstm.weight, -0.1, 0.1)
        lstm.bias = torch.uniform(lstm.bias, -0.1, 0.1)
        enc._rnns[#enc._rnns + 1] = lstm
        if (opt.enc_dropout > 0 and (opt.enc_dropout_loc == 'before' or opt.enc_dropout_loc == 'both')) or
                   (dropout and (opt.dropout_loc == 'before' or opt.dropout_loc == 'both')) then
            enc:add(nn.Dropout(opt.enc_dropout > 0 and opt.enc_dropout  or opt.dropout))
        end
        enc:add(lstm)
        if (opt.enc_dropout > 0 and (opt.enc_dropout_loc == 'after' or opt.enc_dropout_loc == 'both')) or
                (dropout and (opt.dropout_loc == 'after' or opt.dropout_loc == 'both')) then
            enc:add(nn.Dropout(opt.enc_dropout > 0 and opt.enc_dropout  or opt.dropout))
        end
    end
    
    
else
    -- load a model from a th7 file
    enc = torch.load(opt.init_enc_from)
end

if opt.bag_of_words ~= '' then
    local lookup = torch.load(opt.bag_of_words)
    enc = nn.Sequential()
    enc:add(lookup)
    enc:add(nn.Mean(2))
    if not opt.bow_no_linear then
        enc:add(nn.Linear(opt.wordvec_size, opt.hidden_size))
    else
	opt.hidden_size = opt.wordvec_size
    end
    enc:add(nn.Replicate(opt.max_in_len, 2))
    enc._rnns = {}
    dec_lu = lookup:clone('weight', 'gradWeight')
end


-- ***** DECODER *****

if opt.init_dec_from == '' then
    -- Input Layer: Embedding LookupTable. Same as one for encoder so we don't learn two embeddings per word.
    -- we'll be building the decoder with nngraph so we can reuse the lookup layer and pass along hidden state from encoder in training,
    -- since h0 and c0 are graph inputs, we need to make a node for them, done with Identity() layer. nngraph overrides nn.Module()({graph parents})
    local dec_c0 = nn.Identity()()
    local dec_h0 = nn.Identity()()
    local dec_lu = dec_lu()
    dec_rnns = {}

    -- Hidden Layers: N LSTM layers, stacked, with optional dropout. previous helps us form a linear graph with these
    local previous, drop
    previous = {dec_c0, dec_h0, dec_lu}
    for i=1,opt.num_dec_layers do
        local lstm, lstm_n
        if (opt.dec_dropout > 0 and (opt.dec_dropout_loc == 'before' or opt.dec_dropout_loc == 'both')) or
                (dropout and (opt.dropout_loc == 'before' or opt.dropout_loc == 'both')) then
            if i == 1 then
                local drop = nn.ParallelTable()
                drop:add(nn.Dropout(opt.dec_dropout or opt.dropout))
                drop:add(nn.Dropout(opt.dec_dropout or opt.dropout))
                drop:add(nn.Dropout(opt.dec_dropout or opt.dropout))
                local drop_n = drop(previous)
                previous = drop_n
            else
                drop = nn.Dropout(opt.dec_dropout > 0 and opt.dec_dropout or opt.dropout)(previous)
                previous = drop
            end
        end
        if i == 1 then
            if opt.bag_of_words ~= '' and opt.bow_no_linear then
                lstm = nn.LSTM(opt.wordvec_size, opt.wordvec_size)
            else
                lstm = nn.LSTM(opt.wordvec_size, opt.hidden_size)
            end
            lstm.weight = torch.uniform(lstm.weight, -0.1, 0.1)
            lstm.bias = torch.uniform(lstm.bias, -0.1, 0.1)
            lstm_n = lstm(previous)
            previous = lstm_n
        else
            if opt.bag_of_words ~= '' and opt.bow_no_linear then
                lstm = nn.LSTM(opt.wordvec_size, opt.wordvec_size)
            else
                lstm = nn.LSTM(opt.hidden_size, opt.hidden_size)
            end
            lstm.weight = torch.uniform(lstm.weight, -0.1, 0.1)
            lstm.bias = torch.uniform(lstm.bias, -0.1, 0.1)
            lstm_n = lstm(previous)
            previous = lstm_n
        end
        if not opt.dec_forget_states then
            lstm.remember_states = true
        end
        dec_rnns[#dec_rnns + 1] = lstm
        if (opt.dec_dropout > 0 and (opt.dec_dropout_loc == 'after' or opt.dec_dropout_loc == 'both')) or
                (dropout and (opt.dropout_loc == 'after' or opt.dropout_loc == 'both')) then
            drop = nn.Dropout(opt.dec_dropout > 0 and opt.dec_dropout or opt.dropout)(previous)
            previous = drop
        end
    end
    -- now linear transition layers
    local dec_v1, dec_lin
    if opt.bow_no_linear and opt.bag_of_words ~= '' then
        dec_v1 = nn.View(-1, opt.wordvec_size)(previous)
        dec_lin = nn.Linear(opt.wordvec_size, vocab_size)
        dec_lin.weight = torch.uniform(dec_lin.weight, -0.1, 0.1)
        dec_lin.bias = torch.uniform(dec_lin.bias, -0.1, 0.1)
        dec_lin = dec_lin(dec_v1)
    else
        dec_v1 = nn.View(-1, opt.hidden_size)(previous)
        dec_lin = nn.Linear(opt.hidden_size, vocab_size)
        dec_lin.weight = torch.uniform(dec_lin.weight, -0.1, 0.1)
        dec_lin.bias = torch.uniform(dec_lin.bias, -0.1, 0.1)
        dec_lin = dec_lin(dec_v1)
    end

    -- now combine them into a graph module
    dec = nn.gModule({dec_c0, dec_h0, dec_lu}, {dec_lin}) -- {inputs}, {outputs}
    dec._rnns = dec_rnns

else
    -- load a model from a th7 file
    dec = torch.load(opt.init_dec_from)
end

-- =============================================== TRAINING ============================================================

-- Training --
-- We'll use TemporalCrossEntropyCriterion to maximize the likelihood for correct words, ignoring 0 which indicates padding.

criterion = nn.TemporalCrossEntropyCriterion()
local cb
if opt.bag_of_words ~= '' and opt.bow_no_linear then
    cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.wordvec_size)
else
    cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.hidden_size)
end

-- reinforcement learning rate tracking
if opt.track_reinforcement_reward_every > 0 then
    learning_rates = {}
    rewards = {}
end

local hzeros = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.max_in_len-1, opt.hidden_size)
local w0 = torch.CudaTensor({beg_num})
print(beg_num)
print(w0:size())
w0 = w0:expand(opt.batch_size):reshape(opt.batch_size, 1)
-- logging
if opt.save_model_at_epoch then
    logger = optim.Logger(opt.save_prefix .. '.log')
    local names = {'Epoch','Training Loss.', 'Learning Rate:  ', 'T. Perplexity. ', 'V. Loss', 'V. Perplexity'}
    logger:setNames(names)
    logger:display(false) -- prevents display on remote hosts
    logger:style{'+-'} -- points and lines for plot
end

if opt.gpu then
    criterion = criterion:cuda()
    enc = enc:cuda()
    dec = dec:cuda()
end

local params, gradParams = combine_all_parameters(enc, dec)
print('params\n', params:size())
print('gradParams\n', gradParams:size())
local batch = 1
local epoch = 0
local embs
local loss_this_epoch = 0
local perp_this_epoch = 0
local v_loss, v_perp, prev_v_loss, num_increasing_v_loss
num_increasing_v_loss = 0

local euclid = nn.Euclidean(opt.hidden_size, 3):cuda()
local cosine = nn.Cosine(opt.hidden_size, 3):cuda()

-- encoder hidden state monitoring
local final_hidden_states = torch.CudaTensor(enc_inputs:size(1), enc_inputs:size(2), opt.hidden_size)
local e_means = {}; local e_stdevs = {}; local c_means = {}; local c_stdevs = {}
local ek_means = {}; local ek_stdevs = {}; local ck_means = {}; local ck_stdevs = {}

-- output monitoring
-- TODO decide what size to use, etc.
if opt.monitor_outputs then
    -- massive tensor so only creating if used
    output_vectors = torch.CudaTensor(outputs:size(1), outputs:size(2), 5, opt.hidden_size)
end
local gen_e_means = {}; local gen_e_stdevs = {}; local gen_c_means = {}; local gen_c_stdevs = {}
local gen_ek_means = {}; local gen_ek_stdevs = {}; local gen_ck_means = {}; local gen_ck_stdevs = {}
local function print_info(learningRate, iteration, currentError, v_loss, v_perp, t_perp, hidden_states)
    -- TODO move this somewhere else, repeat for generations
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", learningRate)
    if opt.save_model_at_epoch then
        pcall(torch.save, opt.save_prefix..'_enc.th7', enc)
        pcall(torch.save, opt.save_prefix..'_dec.th7', dec)
        local log_result
        log_result = {epoch, currentError, learningRate, t_perp, v_loss, v_perp}
        print(logger.names)
        print(log_result)
        logger:add(log_result)
        if (opt.backup_save_dir ~= '') then 
            pcall(torch.save, opt.backup_save_dir..opt.save_prefix..'_enc.th7', enc)
            pcall(torch.save, opt.backup_save_dir..opt.save_prefix..'_dec.th7', dec)
        end   
    end
end

local optim_config = {learningRate = learningRate }
local avg_diff = 0
local num_diffs = 0

local soft_max = nn.Sequential()
soft_max:add(nn.SoftMax()) -- a little slower but we'll be sampling from it w/ multinomial in reinforcement setting
soft_max:cuda()


local function crossentropy_eval(params)
    gradParams:zero()
    for _, v in pairs(enc._rnns) do  v:resetStates() end
    for _, v in pairs(dec._rnns) do  v:resetStates() end

    -- retrieve inputs for this batch
    local enc_input = enc_inputs[batch]
    local dec_input = dec_inputs[batch]
    local output = outputs[batch]

    -- forward pass
    local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
    local dec_h0 = enc_fwd[{{}, opt.max_in_len, {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len
    if opt.monitor_enc_states then
        final_hidden_states[batch] = dec_h0
    end
    local dec_fwd = dec:forward({cb:clone(), dec_h0, dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
    dec_fwd = torch.reshape(dec_fwd, opt.batch_size, opt.max_out_len, vocab_size)
    if opt.monitor_outputs then
        local dec_hiddens_states = dec._rnns[1].output
        output_vectors[batch] = dec_hiddens_states:sub(1, dec_hiddens_states:size(1), 1, 5):double()
    end
    local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
    _, embs = torch.max(dec_fwd, 3)
    embs = torch.reshape(embs, opt.batch_size, opt.max_out_len)

    -- backward pass
    local cgrd = criterion:backward(dec_fwd, output)
    cgrd = torch.reshape(cgrd, opt.batch_size*opt.max_out_len, vocab_size)
    local hlgrad, dgrd = table.unpack(dec:backward({dec_h0, dec_input}, cgrd))
    local hlgrad = torch.reshape(hlgrad, opt.batch_size, 1, opt.hidden_size)
    local hgrad = torch.cat(hzeros, hlgrad, 2)
    local egrd = enc:backward(enc_input, hgrad)

    --update batch/epoch
    if batch == enc_inputs:size(1) then
        batch = 1
        epoch = epoch + 1
    else
        batch = batch + 1
    end


    return loss, gradParams
end

function run_one_batch(algorithm, optim_config)
    if algorithm == 'adam' then
        return optim.adam(crossentropy_eval, params, optim_config)
    else
        return optim.sgd(crossentropy_eval, params, optim_config)
    end
end

local function reinforcement_eval(params)
    -- TODO: Implement as follows
    -- Choose a particular sample in the mini-batch at random to run the reward program on
    -- Generate a sample using distribution sampling
    gradParams:zero()
    for _, v in pairs(enc._rnns) do  v:resetStates() end
    for _, v in pairs(dec._rnns) do  v:resetStates() end

    -- retrieve inputs for this batch
    local enc_input = enc_inputs[batch]
    local dec_input = dec_inputs[batch]
    local output = outputs[batch]
    local batch_index = torch.random(opt.batch_size)
    local r = reward_maker(output[batch_index], wmap['<mask>']) -- returns a reward function based on output/target

    -- generate first sample
    local enc_fwd = enc:forward(enc_input) -- enc_fwd is h1...hN
    local dec_h0 = enc_fwd[{{}, opt.max_in_len, {}}] -- grab the last hidden state from the encoder, which will be at index max_in_len

    local s1, s2, r1, r2, p1, p2, m1, m2  --sentence 1 & 2, reward 1 & 2, probabilities 1 & 2
    local diff = 0
    local resamples = 0
    local function get_sample_output(r, dec_h0, batch_index)
        local sentence = torch.zeros(opt.max_out_len):cuda()
        local prob = 0
        local cell = cb:clone()
        local hidden = dec_h0:clone()
        local word = w0:clone() -- N x 1
        local cur_dec_in = {cell, hidden, word}
        for t = 1, opt.max_out_len do
            local dec_fwd = dec:forward(cur_dec_in)
            local smax_fwd = soft_max:forward(dec_fwd)
            local w  = torch.multinomial(smax_fwd[batch_index], 1)[1]
            sentence[t] = w
            prob = prob * smax_fwd[batch_index][w]
            if w == end_num then
                break
            end
            -- just using remember_states. Specifying forget_states for enc/dec will cause a logical error
            word = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, 1)
            word[1][1] = w
            cur_dec_in = {cell, hidden, word}
        end
        sentence[1] = beg_num
        local reward, match = r(sentence)
        return sentence, reward, prob, match
    end
    s1, r1, p1, m1 = get_sample_output(r, dec_h0, batch_index)
    while not s2 or diff < opt.reward_threshold do
        s2, r2, p2, m2 = get_sample_output(r, dec_h0, batch_index)
        diff = math.abs(r1 - r2)
        resamples = resamples + 1
        if resamples > opt.max_resample_count then break end
    end
    avg_diff = ((avg_diff * num_diffs) + diff) / (num_diffs + 1)
    num_diffs = num_diffs + 1
    if r2 > r1 then
        local temp
        temp = s1; s1 = s2; s2 = temp
        temp = r1; r1 = r2; r2 = temp
        temp = p1; p1 = p2; p2 = temp
        temp = m1; m1 = m2; m2 = temp
    end

    --adjust learning rate by reward
    local dynamic_learning_rate
    if resamples <= opt.max_resample_count then
            print('changing learning rate: ', optim_config.learningRate, diff, avg_diff)
	    dynamic_learning_rate = learning_rate(optim_config.learningRate, diff, avg_diff)
    else
        dynamic_learning_rate = optim_config.learningRate
    end
    if (batch % opt.track_reinforcement_reward_every == 0) then
        print("Reward Difference: "..math.abs(r2 - r1).."  P1: "..p1.."  Dynamic Learning Rate: "..dynamic_learning_rate)
        if m1 ~= nil and diff > opt.reward_threshold then print(sequence_to_string(m1)) end
        learning_rates[#learning_rates + 1] = dynamic_learning_rate
        rewards[#rewards + 1] = r2 - r1
    end

    -- run the batch on the normal criterion with adjusted learning rate, then revert LR
    local past_lr = optim_config.learningRate
    optim_config.learningRate = dynamic_learning_rate
    local _, loss =  run_one_batch(opt.algorithm, optim_config)
    optim_config.learningRate = past_lr
    return _, loss
end

function train_model()
    while (epoch < opt.max_epochs) do
        local examples = (batch-1)*opt.batch_size
        local output = outputs[batch]
        local out_length = out_lengths[batch]
        local in_length = nl2
        local loss, _
        if opt.reinforcement and epoch >= opt.first_reinforce_epoch then
            _, loss = reinforcement_eval(params)
        else
            _, loss = run_one_batch(opt.algorithm, optim_config)
        end
        local normed_loss = loss[1] / (torch.sum(out_length) / enc_inputs[batch]:size(1))
        loss_this_epoch = loss_this_epoch + (normed_loss / enc_inputs:size(1))
        perp_this_epoch = perp_this_epoch + (torch.exp(normed_loss) / enc_inputs:size(1))

        if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss_this_epoch) end

        -- print info
        if (batch == 1) then
            if (epoch % opt.valid_loss_every == 0) then
                prev_v_loss = v_loss
                v_loss, v_perp  = get_validation_loss(valid_enc_inputs, valid_dec_inputs, valid_outputs, valid_in_lengths, valid_out_lengths)
                if opt.monitor_enc_states and final_hidden_states ~= nil then
                    local hidden_states = final_hidden_states:reshape(enc_inputs:size(1) * enc_inputs:size(2), opt.hidden_size)
                    local e_distances = euclidean_distance(hidden_states, 1)
                    local c_distances = cosine_distance(hidden_states, 1)
                    local e_k = euclid:forward(hidden_states)
                    local c_k = cosine:forward(hidden_states)
                    e_means[#e_means + 1] = e_distances:mean(); e_stdevs[#e_stdevs + 1] = e_distances:std()
                    c_means[#c_means + 1] = c_distances:mean(); c_stdevs[#c_stdevs + 1] = c_distances:std()
                    ek_means[#ek_means + 1] = e_k:mean(); ek_stdevs[#ek_stdevs + 1] = e_k:std()
                    ck_means[#ck_means + 1] = c_k:mean(); ck_stdevs[#ck_stdevs + 1] = c_k:std()
                    local f = io.open(opt.save_prefix..'_hidden_state_distances.json', 'w+')
                    local x = {
                        euclid_distance_means=e_means, euclid_distance_stdevs=e_stdevs,
                        cosine_distance_means=c_means, cosine_distance_stdevs=c_stdevs,
                        euclid_k_distance_means=ek_means, euclid_k_distance_stdevs=ek_stdevs,
                        cosine_k_distance_means=ck_means, cosine_k_distance_stdevs=ck_stdevs
                    }
                    f:write(cjson.encode(x))
                    f:close()
                end
                if opt.monitor_outputs and output_vectors ~= nil then
                    local euc = nn.Euclidean(5 * opt.hidden_size, 3):cuda()
                    local cos = nn.Cosine(5 * opt.hidden_size, 3):cuda()
                    local o_vecs = output_vectors:reshape(outputs:size(1) * outputs:size(2), 5 * opt.hidden_size)
                    local e_distances = euclidean_distance(o_vecs, 1)
                    local c_distances = cosine_distance(o_vecs, 1)
                    local e_k = euc:forward(o_vecs)
                    local c_k = cos:forward(o_vecs)
                    gen_e_means[#gen_e_means + 1] = e_distances:mean(); gen_e_stdevs[#gen_e_stdevs + 1] = e_distances:std()
                    gen_c_means[#gen_c_means + 1] = c_distances:mean(); gen_c_stdevs[#gen_c_stdevs + 1] = c_distances:std()
                    gen_ek_means[#gen_ek_means + 1] = e_k:mean(); gen_ek_stdevs[#gen_ek_stdevs + 1] = e_k:std()
                    gen_ck_means[#gen_ck_means + 1] = c_k:mean(); gen_ck_stdevs[#gen_ck_stdevs + 1] = c_k:std()
                    local f = io.open(opt.save_prefix..'_output_distances.json', 'w+')
                    local x = {
                        euclid_distance_means=gen_e_means, euclid_distance_stdevs=gen_e_stdevs,
                        cosine_distance_means=gen_c_means, cosine_distance_stdevs=gen_c_stdevs,
                        euclid_k_distance_means=gen_ek_means, euclid_k_distance_stdevs=gen_ek_stdevs,
                        cosine_k_distance_means=gen_ck_means, cosine_k_distance_stdevs=gen_ck_stdevs
                    }
                    f:write(cjson.encode(x))
                    f:close()
                end
            end
            print_info(optim_config.learningRate, epoch, loss_this_epoch, v_loss, v_perp, perp_this_epoch, final_hidden_states:reshape(enc_inputs:size(1) * enc_inputs:size(2), opt.hidden_size))
            if v_loss >= prev_v_loss then
                num_increasing_v_loss = num_increasing_v_loss + 1
                if opt.stop_criteria_num_epochs > 0 and num_increasing_v_loss >= opt.stop_criteria_num_epochs then
                    print("Stoppage Criteria met. Stopping after " .. num_increasing_v_loss .. " epochs of increasing validation loss")
                    break
                end
            else
                num_increasing_v_loss = 0
            end
            loss_this_epoch = 0.0
            perp_this_epoch = 0.0

            if data_config ~= nil and input_section < #data_config.enc_inputs then
                input_section = input_section + 1
                enc_inputs = torch.load(data_config.enc_inputs[input_section])
                dec_inputs = torch.load(data_config.dec_inputs[input_section])
                outputs = torch.load(data_config.outputs[input_section])
                in_lengths = torch.load(data_config.in_lengths[input_section])
                out_lengths = torch.load(data_config.out_lengths[input_section])
            elseif data_config ~= nil then
                input_section = 1
                enc_inputs = torch.load(data_config.enc_inputs[input_section])
                dec_inputs = torch.load(data_config.dec_inputs[input_section])
                outputs = torch.load(data_config.outputs[input_section])
                in_lengths = torch.load(data_config.in_lengths[input_section])
                out_lengths = torch.load(data_config.out_lengths[input_section])
            end
        end

        -- print accuracy (handled here so we don't have to pass dec_fwd/embs out of feval)
        if batch % opt.print_acc_every == 0 then
            local acc, nwords = 0, 0
            for n = 1, opt.batch_size do
                nwords = nwords + out_length[n]
                for t = 1, out_length[n] do
                    if embs[n][t] == output[n][t] then
                        acc = acc + 1
                    end
                end
            end
            acc = acc / nwords
            print('Accuracy: ', acc)
        end

        -- print examples
        if batch % opt.print_examples_every == 0 then
            local enc_input = enc_inputs[batch]
            local dec_input = dec_inputs[batch]
            local output = outputs[batch]
            local closs = 0
            for i = 1, opt.batch_size do
                io.write('Encoder Input: ')
                for j = 1, opt.max_in_len do
                    io.write(wmap[enc_input[i][j]] .. ' ')
                end
                print('')
                io.write('Decoder Input: ')
                for j =1, opt.max_out_len do
                    io.write(wmap[dec_input[i][j]] .. ' ')
                end
                print('')
                io.write('Decoder Output: ')
                for j = 1, opt.max_out_len do
                    io.write(wmap[embs[i][j]] .. ' ')
                end
                print('')
                io.write('Ground Truth: ')
                for j = 1, opt.max_out_len do
                    io.write(wmap[output[i][j]] .. ' ')
                end
                print('')
                print('***********')
            end
            print('------------------')
        end
    end
end

function get_validation_loss(venc_inputs, vdec_inputs, voutputs, vin_lengths, vout_lengths)
    enc:evaluate()
    dec:evaluate()
    local v_perp, v_loss = perplexity_over_dataset(enc, dec, venc_inputs, vdec_inputs, vin_lengths, vout_lengths, voutputs)
    enc:training()
    dec:training()
    return v_loss, v_perp
end

if (opt.valid_loss_every > 0) then
    if data_config ~= nil then
        valid_enc_inputs = torch.load(data_config.valid_enc_inputs)
        valid_dec_inputs = torch.load(data_config.valid_dec_inputs)
        valid_outputs = torch.load(data_config.valid_outputs)
        valid_in_lengths = torch.load(data_config.valid_in_lengths)
        valid_out_lengths = torch.load(data_config.valid_out_lengths)
    else
        valid_enc_inputs = torch.load(opt.valid_enc_inputs)
        valid_dec_inputs = torch.load(opt.valid_dec_inputs)
        valid_outputs = torch.load(opt.valid_outputs)
        valid_in_lengths = torch.load(opt.valid_in_lengths)
        valid_out_lengths = torch.load(opt.valid_out_lengths)
    end
    v_loss, v_perp = get_validation_loss(valid_enc_inputs, valid_dec_inputs, valid_outputs, valid_in_lengths, valid_out_lengths)
end

if opt.run then
    train_model()
end

if opt.track_reinforcement_reward_every > 0 then
    local f = io.open(opt.save_prefix..'_rewards.json', 'w+')
    f:write(cjson.encode({rewards=rewards, learning_rates=learning_rates}))
    f:close()
end
