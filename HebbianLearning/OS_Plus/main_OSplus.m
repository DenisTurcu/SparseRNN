function main_OSplus(n_iter, PN1, PN2, P_step, N_min, N_max, N_step, sparsity, g_factor, n_epochs)
%main_OSplus Generate the data to be plotted for the  results
%   Run the OS+ learning with Hopfield J for multiple pairs (N,P) and
%   record accuracy for each of them. 

% generate the (N,P) pairs
Ns = N_min:N_step:N_max;
N_all = [];
P_all = [];
for i = 1:length(Ns)
    P_min = floor(1/PN1 * sparsity * Ns(i)^2 * (2*sparsity));
    P_max = floor(1/PN2 * sparsity * Ns(i)^2 * (2*sparsity));
    Ps = P_min:P_step:P_max;
    N_all = [N_all, Ns(i) * ones(1,length(Ps))];
    P_all = [P_all, Ps];
end

% store results
results = zeros(length(N_all), n_iter);

for i=1:n_iter
    disp(i);
    tic;
    % for each iteration, run the simulation (in parralel for speed)
    parfor j=1:length(N_all)
        results(j,i) = OSplus_SRNN(N_all(j), P_all(j), sparsity, sparsity, g_factor, n_epochs);
    end
    toc;
    
    % save results
    save('ReplaceWithDate_results_OSplus.mat','results','N_all','P_all');
end
end

% PN1 = 18; PN2 = 7; P_step = 40; N_min = 250; N_max = 1000; N_step = 25; sparsity = 0.1; g_factor = 10; n_epochs = 150;