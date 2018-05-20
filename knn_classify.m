function knn_classify(training_file, test_file, k)

    train_set = importdata(training_file);
    test_set = importdata(test_file);
    k = str2num(k);
    
    train_class = train_set(:, end);
    test_class = test_set(:, end);
    train_set = train_set(:, 1:end-1);
    test_set = test_set(:,1:end-1);
    
    m = mean(train_set);
    sd = std(train_set,1);
    train_set = (train_set - m)./sd;
    test_set = (test_set - m)./sd;
    accuracy = [];
    for test_row = 1:size(test_set,1)
        dist = (sum((train_set - test_set(test_row,:)).^2,2)).^0.5;
        [~,mindist_index] = sortrows(dist);
        neighbr = train_class(mindist_index(1:k));
        nb_cl = unique(neighbr);
        if length(nb_cl) == 1
            predicted_class = nb_cl;
            if nb_cl == test_class(test_row)
                accu = 1;
            else
                accu = 0;
            end
        else
            cl_hist = histcounts(neighbr, length(nb_cl));
            pred = nb_cl(find(cl_hist == max(cl_hist)));
            if length(pred) == 1
                predicted_class = pred;
                if pred == test_class(test_row)
                    accu = 1;
                else
                    accu = 0;
                end
            else
                predicted_class = datasample(pred,1);
                if find(pred == test_class(test_row))
                    accu = 1/length(pred);
                else
                    accu = 0;
                end
            end
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', test_row-1, predicted_class, test_class(test_row), accu);
        accuracy = [accuracy, accu];
    end
    fprintf('classification accuracy=%6.4f\n', mean(accuracy));
end

