--
-- Created by IntelliJ IDEA.
-- User: bking
-- Date: 1/26/17
-- Time: 2:00 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'
require 'math'

-- This class represents a sampling layer. Ite expects vectors representing a distribition (MUST sum to 1)

local layer, parent = torch.class('nn.Sampler', 'nn.Module')

function layer:__init()
    parent.__init(self)
    self.argmax = true
end

function layer:updateOutput(input)
    if self.argmax then
        self.output, self.prob_values = self:argMax(input)
    else
        self.output, self.prob_values = self:sampleDistribution(input)
    end
    return self.output
end

function layer:sampleDistribution(input)
    -- Is this the same as torch.multinomial? Can this be replaced/re-implemented?
    local sample = torch.IntTensor(input:size()[1])
    local prob_values = torch.DoubleTensor(input:size()[1])
    for i=1,input:size()[1] do
        local in_exp = torch.exp(input[i])
        local sorted, indices = in_exp:sort()
        local sum = 0.0
        local value = math.random()
        for j=1,sorted:size()[1] do
            local index = sorted:size()[1] + 1 - j
            sum = sum + sorted[index]
            if sum > value then
                sample[i] = indices[index]
                prob_values[i] = sorted[index]
                break
            end
        end
    end
    return sample, prob_values
end

function layer:argMax(input)
    local sample = torch.IntTensor(input:size()[1])
    local prob_values = torch.DoubleTensor(input:size()[1])
    for i=1,input:size()[1] do
        local in_exp = torch.exp(input[i])
        local max = in_exp[1]
        local argmax = 1
        for j=1,in_exp:size()[1] do
            if in_exp[j] > max then
                max = in_exp[j]
                argmax = j
            end
        end
        sample[i] = argmax
        prob_values[i] = max
    end
    return sample, prob_values
end

function layer:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

function layer:accGradParameters(input, gradOutput)
    return gradOutput
end

function layer:updateParamaters()
    return nil
end

function layer:parameters()
    return nil
end


function layer:training()
    parent.training(self)
end

function layer:evaluate()
    parent.evaluate(self)
end

