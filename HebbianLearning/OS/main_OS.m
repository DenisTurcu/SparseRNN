function main_OS(n_iter, PN1, PN2, P_step, N_min, N_max, N_step, sparsity, g_factor)
%main_OS Generate the data to be plotted for the results
%   Run the OS "learning" with Hopfield J for multiple pairs (N,P) and
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
        results(j,i) = OS_SRNN(N_all(j), P_all(j), sparsity, sparsity, g_factor);
    end
    toc;
    
    % save results
    save('ReplaceWithDate_results.mat','results','N_all','P_all');
end
end

% PN1 = 100; PN2 = 20; P_step = 50; N_min = 500; N_max = 2000; N_step = 50; sparsity = 0.1; g_factor = 10;