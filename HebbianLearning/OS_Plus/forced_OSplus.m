function forced_OSplus(n_iter, Ps, Ns, sparsity, g_factor, n_epochs)
%UNTITLED Generate the data to be plotted for the preliminary results
%   Run the OS+ learning with Hopfield J for multiple pairs (N,P) and
%   record accuracy for each of them. 

% store results
results = zeros(length(Ns), n_iter, n_epochs);

for i=1:n_iter
    disp(i);
    tic;
    % for each iteration, run the simulation (in parralel for speed)
    parfor j=1:length(Ns)
        [~, results(j,i,:)] = OSplus_SRNN(Ns(j), Ps(j), sparsity, sparsity, g_factor, n_epochs);
    end
    toc;
    
    % save results
    save('220810_results_forced_OSplus.mat','results','Ns','Ps');
end
end


