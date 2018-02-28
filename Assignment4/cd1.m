function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    % error('not yet implemented');

    visible_data = sample_bernoulli(visible_data);

    hp = visible_state_to_hidden_probabilities(rbm_w, visible_data); % size <number of hidden units> by <number of configurations that we're handling in parallel>.
    sample_hp = sample_bernoulli(hp);

    d1 = configuration_goodness_gradient(visible_data, sample_hp);

    vp = hidden_state_to_visible_probabilities(rbm_w, sample_hp); %  size <number of visible units> by <number of configurations that we're handling in parallel>.
    sample_vp = sample_bernoulli(vp);

    hp = visible_state_to_hidden_probabilities(rbm_w, sample_vp); % size <number of hidden units> by <number of configurations that we're handling in parallel>.
    sample_hp = sample_bernoulli(hp); % Instead of a sampled state, we'll simply use the conditional probabilities.

    d2 = configuration_goodness_gradient(sample_vp, hp); % use hp instead of sample_hp, reason see README.md, question 8

    ret = d1 - d2;

end
