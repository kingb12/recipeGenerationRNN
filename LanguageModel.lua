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
cmd:option('-dec_inputs', '../data/rl_dec_inputs.th7')
cmd:option('-outputs', '../data/rl_outputs.th7')
cmd:option('-out_lengths', '../data/rl_out_lengths.th7')
cmd:option('-helper', '../data/rl_helper.th7')

cmd:option('-data_config', '', 'load data from a config file. also allows specifying multiple sets')

cmd:option('-valid_dec_inputs', '/homes/iws/kingb12/data/rl_vdec_inputs.th7')
cmd:option('-valid_outputs', '/homes/iws/kingb12/data/rl_voutputs.th7')
cmd:option('-valid_out_lengths', '/homes/iws/kingb12/data/rl_vout_lengths.th7')

cmd:option('-max_out_len', 300, 'max decoder sequence length')
cmd:option('-min_out_len', 1, 'min encoder sequence length')
cmd:option('-min_in_len', 1, 'min decoder sequence length')
cmd:option('-batch_size', 4)
cmd:option('-stop_criteria_num_epochs', 0, 'cutoff for number of epochs of increaseing valid loss after which to stop')

-- Model options
cmd:option('-init_dec_from', '')
cmd:option('-wordvec_size', 100)
cmd:option('-hidden_size', 512)
cmd:option('-dropout', 0)
cmd:option('-dec_dropout', 0)
cmd:option('-dropout_loc', 'after')
cmd:option('-dec_dropout_loc', 'after')


cmd:option('-num_dec_layers', 1)
cmd:option('-weights', '')
cmd:option('-no_average_loss', false)
cmd:option('-dec_forget_states', false)
cmd:option('-bag_of_words', '', 'encoder is replaced with a bag of words approach')
cmd:option('-bow_no_linear', false, 'only meaningful in bag_of_words context. no linear projection to hidden size, all are wordvec_size')

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 0.1)
cmd:option('-lr_decay', 0.0)
cmd:option('-algorithm', 'adam')
cmd:option('-reinforcement', false)
cmd:option('-print_reinforcement_reward_every',0)
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


-- Backend options
cmd:option('-gpu', false)

local opt = cmd:parse(arg)
local tensorType = 'torch.FloatTensor'
local learningRate = opt.learning_rate
local dropout = opt.dropout > 0.0
local pad_num = 1
local beg_num = 2
local end_num = 3

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
    dec_inputs = torch.load(data_config.dec_inputs[1]) -- recipe
    outputs = torch.load(data_config.outputs[1]) -- recipe shifted one over (like a LM)
    out_lengths = torch.load(data_config.out_lengths[1]) -- lengths specifying end of padding
    helper = torch.load(data_config.helper) -- has word_map, reverse, etc.
else
    dec_inputs = torch.load(opt.dec_inputs) -- recipe
    outputs = torch.load(opt.outputs) -- recipe shifted one over (like a LM)
    out_lengths = torch.load(opt.out_lengths) -- lengths specifying end of padding
    helper = torch.load(opt.helper) -- has word_map, reverse, etc.
end
local vocab_size = #helper.n_to_w
local wmap = helper.n_to_w

-- =========================================== THE MODEL ===============================================================

-- ***** ENCODER *****

local lu = nn.LookupTable(vocab_size, opt.wordvec_size)
lu.weight = torch.uniform(lu.weight, -0.1, 0.1)
local dec_lu = lu:clone('weight', 'gradWeight')

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
local cb = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.hidden_size)
local hzeros = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, opt.hidden_size)
local w0 = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, 1)
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
    dec = dec:cuda()
end

local params, gradParams = combine_all_parameters(dec)
local batch = 1
local epoch = 0
local embs
local loss_this_epoch = 0
local perp_this_epoch = 0
local v_loss, v_perp, prev_v_loss, num_increasing_v_loss
num_increasing_v_loss = 0


local function print_info(learningRate, iteration, currentError, v_loss, v_perp, t_perp, hidden_states)
    -- TODO move this somewhere else, repeat for generations
    print("Current Iteration: ", iteration)
    print("Current Loss: ", currentError)
    print("Current Learing Rate: ", learningRate)
    if opt.save_model_at_epoch then
        pcall(torch.save, opt.save_prefix..'_dec.th7', dec)
        local log_result
        log_result = {epoch, currentError, learningRate, t_perp, v_loss, v_perp}
        print(logger.names)
        print(log_result)
        logger:add(log_result)
        if (opt.backup_save_dir ~= '') then 
            pcall(torch.save, opt.backup_save_dir..opt.save_prefix..'_dec.th7', dec)
        end    
    end
end

local optim_config = {learningRate = learningRate }
local avg_diff = 0
local num_diffs = 0

local lsm = nn.Sequential()
lsm:add(nn.View(-1))
lsm:add(nn.LogSoftMax())
lsm:cuda()


local function crossentropy_eval(params)
    gradParams:zero()
    for _, v in pairs(dec._rnns) do  v:resetStates() end

    -- retrieve inputs for this batch
    local dec_input = dec_inputs[batch]
    local output = outputs[batch]

    -- forward pass
    local dec_fwd = dec:forward({cb:clone(), hzeros:clone(), dec_input}) -- forwarding a new zeroed cell state, the encoder hidden state, and frame-shifted expected output (like LM)
    dec_fwd = torch.reshape(dec_fwd, opt.batch_size, opt.max_out_len, vocab_size)
    local loss = criterion:forward(dec_fwd, output) -- loss is essentially same as if we were a language model, ignoring padding
    _, embs = torch.max(dec_fwd, 3)
    embs = torch.reshape(embs, opt.batch_size, opt.max_out_len)

    -- backward pass
    local cgrd = criterion:backward(dec_fwd, output)
    cgrd = torch.reshape(cgrd, opt.batch_size*opt.max_out_len, vocab_size)
    local hlgrad, dgrd = table.unpack(dec:backward({dec_h0, dec_input}, cgrd))

    --update batch/epoch
    if batch == dec_inputs:size(1) then
        batch = 1
        epoch = epoch + 1
    else
        batch = batch + 1
    end

    return loss, gradParams
end

function run_one_batch(algorithm)
    if algorithm == 'adam' then
        return optim.adam(crossentropy_eval, params, optim_config)
    else
        return optim.sgd(crossentropy_eval, params, optim_config)
    end
end

local function reinforcement_eval(params)
    gradParams:zero()
    for _, v in pairs(dec._rnns) do  v:resetStates() end

    -- retrieve inputs for this batch
    local dec_input = dec_inputs[batch]
    local output = outputs[batch]
    local r = reward_maker(output) -- returns a reward function based on output/target

    -- generate first sample
    local dec_h0 = hzeros:clone() -- grab the last hidden state from the encoder, which will be at index max_in_len

    local s1, s2, r1, r2, ll1, ll2  --sentence 1 & 2, reward 1 & 2, log_likelihoods 1 & 2
    local diff = 0
    local resamples = 0

    local function get_sample_output(r, dec_h0)
        local s1 = torch.zeros(1, opt.max_out_len):cuda()
        local ll1 = 0
        local cell = cb:clone()
        local hidden = dec_h0:clone()
        local word = w0:clone()
        local cur_dec_in = {cell, hidden, w0}
        for t = 1, opt.max_out_len do
            local dec_fwd = dec:forward(cur_dec_in)
            local w  = torch.multinomial(dec_fwd[1], 1)[1]
            s1[1][t] = w
            ll1 = ll1 + lsm:forward(dec_fwd)[w]
            if w == end_num then
                break
            end
            -- just using remember_states. Specifying forget_states for enc/dec will cause a logical error
            word = torch.CudaTensor.zeros(torch.CudaTensor.new(), opt.batch_size, 1)
            word[1][1] = w
            cur_dec_in = {cell, hidden, word}
        end
        s1[1][1] = beg_num
        local r1 = r(s1)
        return s1, r1, ll1
    end
    s1, r1, ll1 = get_sample_output(r, dec_h0)
    while not s2 or diff < opt.reward_threshold do
        s2, r2, ll2 = get_sample_output(r, dec_h0)
        diff = math.abs(r1 - r2)
        print(diff)
        resamples = resamples + 1
        if resamples > opt.max_resample_count then break end
    end
    avg_diff = ((avg_diff * num_diffs) + diff) / (num_diffs + 1)
    num_diffs = num_diffs + 1
    if r2 > r1 then
        local temp
        temp = s1; s1 = s2; s2 = temp
        temp = r1; r1 = r2; r2 = temp
        temp = ll1; ll1 = ll2; ll2 = temp
    end

    --adjust learning rate by reward
    local p1 = math.exp(ll1)
    local dynamic_learning_rate
    if resamples <= opt.max_resample_count then
	dynamic_learning_rate = learning_rate(optim_config.learningRate, p1, diff, avg_diff)
    else
        dynamic_learning_rate = optim_config.learningRate
    end
    if (batch % opt.print_reinforcement_reward_every == 0) then
        print("Reward Difference: "..math.abs(r2 - r1).."  P1: "..p1.."  Dynamic Learning Rate: "..dynamic_learning_rate)
    end
    optim_config.learningRate = dynamic_learning_rate

    return run_one_batch(opt.algorithm)
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
            _, loss = run_one_batch(opt.algorithm)
        end
        local normed_loss = loss[1] / (torch.sum(out_length) / dec_inputs[batch]:size(1))
        loss_this_epoch = loss_this_epoch + (normed_loss / dec_inputs:size(1))
        perp_this_epoch = perp_this_epoch + (torch.exp(normed_loss) / dec_inputs:size(1))

        if (batch % opt.print_loss_every) == 0 then print('Loss: ', loss_this_epoch) end

        -- print info
        if (batch == 1) then
            if (epoch % opt.valid_loss_every == 0) then
                prev_v_loss = v_loss
                v_loss, v_perp  = get_validation_loss(valid_dec_inputs, valid_outputs, valid_out_lengths)
            end
            print_info(optim_config.learningRate, epoch, loss_this_epoch, v_loss, v_perp, perp_this_epoch)
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

            if data_config ~= nil and input_section < #data_config.dec_inputs then
                input_section = input_section + 1
                dec_inputs = torch.load(data_config.dec_inputs[input_section])
                outputs = torch.load(data_config.outputs[input_section])
                out_lengths = torch.load(data_config.out_lengths[input_section])
            elseif data_config ~= nil then
                input_section = 1
                dec_inputs = torch.load(data_config.dec_inputs[input_section])
                outputs = torch.load(data_config.outputs[input_section])
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
            local dec_input = dec_inputs[batch]
            local output = outputs[batch]
            local closs = 0
            for i = 1, opt.batch_size do
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

function get_validation_loss(vdec_inputs, voutputs, vout_lengths)
    dec:evaluate()
    local v_perp, v_loss = lm_perplexity_over_dataset(dec, vdec_inputs, vout_lengths, voutputs, opt.hidden_size)
    dec:training()
    return v_loss, v_perp
end

if (opt.valid_loss_every > 0) then
    if data_config ~= nil then
        valid_dec_inputs = torch.load(data_config.valid_dec_inputs)
        valid_outputs = torch.load(data_config.valid_outputs)
        valid_out_lengths = torch.load(data_config.valid_out_lengths)
    else
        valid_dec_inputs = torch.load(opt.valid_dec_inputs)
        valid_outputs = torch.load(opt.valid_outputs)
        valid_out_lengths = torch.load(opt.valid_out_lengths)
    end
    v_loss, v_perp = get_validation_loss(valid_dec_inputs, valid_outputs, valid_out_lengths)
end

if opt.run then
    train_model()
end
