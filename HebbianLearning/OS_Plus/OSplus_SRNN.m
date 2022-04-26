function [accuracy] = OSplus_SRNN(N, P, sparsity, out_sparsity, g_factor, n_epochs)
%OSplus_SRNN Function to run simulation for a single (N,P) pair
%   Just run the simulation for given parameters. This function is used in
%   "main_OSplus" for each iteration and pair (N,P).

eta = g_factor * 0.1^(1/4) / sqrt(N * P * sqrt(sparsity)); % synaptic weight factor
dyn_thresh = 0.0334;
epsilon = 0.03 * log(2);

dt = 0.1;           % time step
T = 2;             % duration of trail
t = 0:dt:T;         % time variable

s = 2*rand(N, P) - 1;       % memory patterns
b = 2*(randn(P, 1)>0) - 1;  % category of patterns

p = zeros(length(t), 1);    % input pulse  
p(1:10) = 1;

M = (sprandn(N, N, sparsity)~=0);  % connectivity mask               
J = eta*(s*diag(b)*s').*M;  % connectivity matrix

w_out = (sprandn(1, N, out_sparsity)~=0);

accuracies = zeros(n_epochs, 1);
xL = 0.5;
for epoch = 1:n_epochs
    pat_ids = randperm(P);
    rE_dyn = zeros(P, 1);
    for j = 1:P                       % loop over patterns
        x = zeros(N, length(t));    % network variable  
        r = zeros(N, length(t));    % network rates
        for i=2:length(t)           % loop over time
            x(:, i) = x(:, i-1)+dt*(-x(:, i-1)+J*r(:, i-1)+p(i)*s(:, pat_ids(j)));
            r(:, i) = 1 * (tanh(max(x(:, i), 0)-xL)-tanh(-xL)) / (1 - tanh(-xL)); % rectify, square and clip
        end        
        rE_dyn(pat_ids(j)) = w_out * r(:, end)/ (out_sparsity * N);  % mean response for current pattern and trial
        % update J if pattern labeled incorrectly
        if (2 * (rE_dyn(pat_ids(j)) > dyn_thresh) - 1) * b(pat_ids(j)) < 0
            J = J + (epsilon / log(1+1)) * eta * b(pat_ids(j)) * (s(:, pat_ids(j)) * s(:, pat_ids(j))' .* M);
        end
    end
%     dyn_thresh = prctile(rE_dyn, 100 * (sum(b==-1)/P));  % use dynamic threshold 
    a_dyn = (2 * (rE_dyn > dyn_thresh) - 1) .* b;
    accuracies(epoch) = 100 * sum(a_dyn>0,1) / P;
end
accuracy = max(accuracies);

end
