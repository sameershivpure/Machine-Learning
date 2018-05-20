function pca_power(training_file, test_file, pca_dimen, iterations)

    train_set = importdata(training_file);
    test_set = importdata(test_file);
    pca_dimen = str2num(pca_dimen);
    iterations = str2num(iterations);

    train_set = train_set(:,1:end-1);
    test_set = test_set(:,1:end-1);
    eg_vec = zeros(pca_dimen, size(train_set,2));
    
    data_set = train_set;
    for vec_in = 1:pca_dimen
        data_cov = cov(data_set);
        vect = pow_method(data_cov, iterations);
        eg_vec(vec_in,:) = vect;
        data_set = data_set - ((vect * data_set')' * vect);
        
        fprintf('Eigenvector %d\n', vec_in);
        for i = 1:length(vect)
           fprintf('%3d: %.4f\n', i, vect(i)); 
        end
        fprintf('\n');
    end
    
    for test_row = 1:size(test_set,1)
        comp = eg_vec*test_set(test_row,:)';
        fprintf('Test object %d\n', test_row-1);
        for index = 1:length(comp)
           fprintf('%3d: %.4f\n', index, comp(index));
        end
    end
    
end

function [vect] = pow_method(cov_mat, itr)
    
    b = rand(size(cov_mat,2),1);
    for r = 1:itr
        b = cov_mat*b;
        b = b/norm(b);
    end
    vect = b';
end

