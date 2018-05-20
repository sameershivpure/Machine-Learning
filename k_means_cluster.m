function k_means_cluster(test_file, k, iteration)

    data_set = importdata(test_file);
    k = str2num(k);
    iteration = str2num(iteration);
    
    data_set(:,end) = randi(k,[size(data_set,1),1]);
    
    for iter = 1:iteration+1
        cluster_mean = [];
        error = 0;
        for k_in = 1:k
            m = mean(data_set(data_set(:,end) == k_in,1:end-1));
            cluster_mean = [cluster_mean;m];   
            error = error + sum(sqrt(sum((data_set(data_set(:,end) == k_in,1:end-1)-m).^2,2)));
        end
        if iter == 1
            fprintf('After initialization: error = %.4f\n',error);
        else
            fprintf('After iteration %d: error = %.4f\n',iter-1, error);
        end
        for test_row = 1:size(data_set,1)
           [~,data_set(test_row,end)] = min(sqrt(sum((cluster_mean - data_set(test_row,1:end-1)).^2,2)));
        end
    end

end

