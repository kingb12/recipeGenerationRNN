--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 2/19/17
-- Time: 12:10 PM
-- To change this template use File | Settings | File Templates.
--

package.path = ';/homes/iws/kingb12/LanguageModelRNN/?.lua;'..package.path

require 'torch'
require 'nn'
require 'nnx'
require 'nngraph'
require 'util'
require 'DynamicView'
require 'TemporalCrossEntropyCriterion'
require 'LSTM'
require 'Sampler'
cjson = require 'cjson'
require 'cutorch'
require 'cunn'
require 'encdec_eval_functions'

torch.setheaptracking(true)

-- =========================================== COMMAND LINE OPTIONS ====================================================

local cmd = torch.CmdLine()
-- Options
cmd:option('-calculate_perplexity', false)
cmd:option('-calculate_bleu', false)
cmd:option('-generate_samples', false)
cmd:option('-calculate_avg_alignment', false)
cmd:option('-calculate_n_pairs_bleu', false)
cmd:option('-calculate_best_bleu_match', false)
cmd:option('-count_ingredients_used', false)




-- Dataset options
cmd:option('-train_enc_inputs', '/homes/iws/kingb12/data/rl_enc_inputs.th7')
cmd:option('-train_dec_inputs', '/homes/iws/kingb12/data/rl_dec_inputs.th7')
cmd:option('-train_outputs', '/homes/iws/kingb12/data/rl_outputs.th7')
cmd:option('-train_in_lengths', '/homes/iws/kingb12/data/rl_in_lengths.th7')
cmd:option('-train_out_lengths', '/homes/iws/kingb12/data/rl_out_lengths.th7')

cmd:option('-valid_enc_inputs', '/homes/iws/kingb12/data/rl_venc_inputs.th7')
cmd:option('-valid_dec_inputs', '/homes/iws/kingb12/data/rl_vdec_inputs.th7')
cmd:option('-valid_outputs', '/homes/iws/kingb12/data/rl_voutputs.th7')
cmd:option('-valid_in_lengths', '/homes/iws/kingb12/data/rl_vin_lengths.th7')
cmd:option('-valid_out_lengths', '/homes/iws/kingb12/data/rl_vout_lengths.th7')

cmd:option('-test_enc_inputs', '/homes/iws/kingb12/data/rl_tenc_inputs.th7')
cmd:option('-test_dec_inputs', '/homes/iws/kingb12/data/rl_tdec_inputs.th7')
cmd:option('-test_outputs', '/homes/iws/kingb12/data/rl_toutputs.th7')
cmd:option('-test_in_lengths', '/homes/iws/kingb12/data/rl_tin_lengths.th7')
cmd:option('-test_out_lengths', '/homes/iws/kingb12/data/rl_tout_lengths.th7')

cmd:option('-helper', '../data/rl_helper.th7')
cmd:option('-init_output_from', '', 'useful for updating report without doing intensive things like generating samples')

-- Model options
cmd:option('-enc', 'enc.th7')
cmd:option('-dec', 'dec.th7')

--Output Options
cmd:option('-num_samples', 10)
cmd:option('-max_sample_length', 10)
cmd:option('-max_gen_example_length', 10)
cmd:option('-gen_inputs', '', 'A String as JSON encoded list of integers of which training samples to use for generations. For better comparing N models')
cmd:option('-ingredient_inputs', '', 'A JSON file containing a list ingredients to count against for each generation. Only makes sense when gen_inputs supplied and you list ongredients for these recipes in the file')
cmd:option('-no_arg_max', false)
cmd:option('-out', '')

local opt = cmd:parse(arg)
-- ================================================ EVALUATION =========================================================

train_enc_inputs = torch.load(opt.train_enc_inputs)
train_dec_inputs = torch.load(opt.train_dec_inputs)
train_outputs = torch.load(opt.train_outputs)
train_in_lengths = torch.load(opt.train_in_lengths)
train_out_lengths = torch.load(opt.train_out_lengths)

valid_enc_inputs = torch.load(opt.valid_enc_inputs)
valid_dec_inputs = torch.load(opt.valid_dec_inputs)
valid_outputs = torch.load(opt.valid_outputs)
valid_in_lengths = torch.load(opt.valid_in_lengths)
valid_out_lengths = torch.load(opt.valid_out_lengths)

test_enc_inputs = torch.load(opt.test_enc_inputs)
test_dec_inputs = torch.load(opt.test_dec_inputs)
test_outputs = torch.load(opt.test_outputs)
test_in_lengths = torch.load(opt.test_in_lengths)
test_out_lengths = torch.load(opt.test_out_lengths)


enc = torch.load(opt.enc)
dec = torch.load(opt.dec)
helper = torch.load(opt.helper)

enc:evaluate()
dec:evaluate()

for k,v in pairs(enc._rnns) do v.remember_states = false end
for k,v in pairs(dec._rnns) do v.remember_states = false end

criterion = nn.TemporalCrossEntropyCriterion():cuda()

-- We will build a report as a table which will be converted to json.
if opt.init_output_from ~= nil and opt.init_output_from ~= '' then
    local f = io.open(opt.init_output_from, 'r')
    output = cjson.decode(f:read())
    f:close()
else
    output = {}
end

sampler =  nn.Sampler():cuda()
if opt.no_arg_max then
    sampler.argmax = false
end

if opt.calculate_perplexity then
    print('Calculating Training Perplexity...')
    local tr_perp = perplexity_over_dataset(enc, dec, train_enc_inputs, train_dec_inputs, train_in_lengths, train_out_lengths, train_outputs)
    output['train_perplexity'] = tr_perp
    print('Calculating Valid Perplexity...')
    local vd_perp = perplexity_over_dataset(enc, dec, valid_enc_inputs, valid_dec_inputs, valid_in_lengths, valid_out_lengths, valid_outputs)
    output['valid_perplexity'] = vd_perp
    print('Calculating Test Perplexity...')
    local ts_perp = perplexity_over_dataset(enc, dec, test_enc_inputs, test_dec_inputs, test_in_lengths, test_out_lengths, test_outputs)
    output['test_perplexity'] = ts_perp
end

-- generate some samples
if opt.generate_samples then
    local gen_inputs = nil
    local ingredients = nil
    if opt.gen_inputs ~= '' then
        gen_inputs = cjson.decode(opt.gen_inputs)
    end
    if opt.count_ingredients_used ~= '' then
        if opt.gen_inputs == '' then
            print('Warning: Random gnerations used with particular ingredient counts')
        end
        local io = require 'io'
        local f = io.open(opt.count_ingredients_used, 'r')
        ingredients = cjson.decode(f:read())
    end
    output['train_samples'] = generate_samples(train_enc_inputs, train_outputs, opt.num_samples, opt.max_sample_length, gen_inputs, ingredients)
    output['valid_samples'] = generate_samples(valid_enc_inputs, valid_outputs, opt.num_samples, opt.max_sample_length, gen_inputs, ingredients)
    output['test_samples'] = generate_samples(test_enc_inputs, test_outputs, opt.num_samples, opt.max_sample_length, gen_inputs, ingredients)
end

if opt.calculate_bleu or opt.calculate_n_pairs_bleu or opt.calculate_avg_alignment then
    references = {}
    candidates = {}
    tr_references = {}
    tr_candidates = {}
    print(tr_candidates)
    v_references = {}
    v_candidates = {}
    ts_references = {}
    ts_candidates = {}
    for i=1,#output['train_samples'] do
        tr_candidates[#tr_candidates + 1] = output['train_samples'][i]['generated']
        tr_references[#tr_references + 1] = output['train_samples'][i]['gold']
        candidates[#candidates + 1] = output['train_samples'][i]['generated']
        references[#references + 1] = output['train_samples'][i]['gold']
    end
    for i=1,#output['valid_samples'] do
        v_candidates[#v_candidates + 1] = output['valid_samples'][i]['generated']
        v_references[#v_references + 1] = output['valid_samples'][i]['gold']
        candidates[#candidates + 1] = output['valid_samples'][i]['generated']
        references[#references + 1] = output['valid_samples'][i]['gold']
    end
    for i=1,#output['test_samples'] do
        ts_candidates[#ts_candidates + 1] = output['test_samples'][i]['generated']
        ts_references[#ts_references + 1] = output['test_samples'][i]['gold']
        candidates[#candidates + 1] = output['test_samples'][i]['generated']
        references[#references + 1] = output['test_samples'][i]['gold']
    end
    output['train_bleu'] = calculate_bleu(tr_references, tr_candidates)
    output['valid_bleu'] = calculate_bleu(v_references, v_candidates)
    output['test_bleu'] = calculate_bleu(ts_references, ts_candidates)
    output['combined_bleu'] = calculate_bleu(references, candidates)
end

if opt.calculate_n_pairs_bleu then
    print("N pairs BLEU...")
    output['n_pairs_bleu_train'] = n_pairs_bleu(tr_candidates, 1000)
    output['n_pairs_bleu_valid'] = n_pairs_bleu(v_candidates, 1000)
    output['n_pairs_bleu_test'] = n_pairs_bleu(ts_candidates, 1000)
    output['n_pairs_bleu_all'] = n_pairs_bleu(candidates, 1000)
    output['n_pairs_bleu_gold'] = n_pairs_bleu(references, 1000)
end

if opt.calculate_avg_alignment then
    print("Average alignment...")
    output['average_alignment_train'] = alignment_scores(tr_candidates)
    output['average_alignment_valid'] = alignment_scores(v_candidates)
    output['average_alignment_test'] = alignment_scores(ts_candidates)
    output['average_alignment_all'] = alignment_scores(candidates)
    output['average_alignment_gold'] = alignment_scores(references)
end

if opt.calculate_best_bleu_match then
    print("Best Bleu Matches....")
    local full_train_set = {}
    for i=1, train_outputs:size(1) do for j=1, train_outputs:size(2) do
        full_train_set[#full_train_set + 1] = sequence_to_string(train_outputs[i][j])
    end end
    local function extract_generations(gen_struct)
        local result = {}
        for i=1, #gen_struct do
            result[#result + 1] = gen_struct[i]['generated']
        end
        return result
    end
    local function get_best_matches(refs, gens)
        local result = {}
        for i=1, #gens do
            local r = {}
            local best_match, best_score = closest_bleu_match(refs, gens[i])
            r['best_match'] = best_match
            r['best_score'] = best_score
            result[#result + 1] = r
        end
        return result
    end
    print("    Train....")
    output['best_bleu_matches_train'] = closest_bleu_match(full_train_set, extract_generations(output['train_samples']))
    print("    Valid....")
    output['best_bleu_matches_valid'] = closest_bleu_match(full_train_set, extract_generations(output['valid_samples']))
    print("    Test....")
    output['best_bleu_matches_test'] = closest_bleu_match(full_train_set, extract_generations(output['test_samples']))
end

output['architecture'] = {}
output['architecture']['encoder'] = tostring(enc)
output['architecture']['decoder'] = tostring(dec)

if opt.out ~= '' then
    local s = cjson.encode(output)
    local io = require 'io'
    local f = io.open(opt.out, 'w+')
    f:write(s)
    f:close()
end
