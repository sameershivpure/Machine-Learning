function dtree_classify( test_data, class_set, d_tree)
       t_accu = [];
    fprintf('\n Classification result\n');
    for test_row = 1:size(test_data,1)
        dist = zeros(size(d_tree,2),length(class_set));
        for tree_in = 1:length(d_tree)
           cur_node = d_tree(tree_in);
           leaf = 0;
           while ~leaf
               if strcmpi(class(cur_node),'struct')
                   if test_data(test_row, cur_node.('attr')) < cur_node.('thres')
                      cur_node =  cur_node.('left_child');
                   else
                      cur_node =  cur_node.('right_child');
                   end
               else
                dist(tree_in,:) = cur_node;
                leaf = 1;
               end

           end
        end
        avg_dist = mean(dist,1);
        max_prob = max(avg_dist);
        max_indexes = find(avg_dist == max_prob);
        if (length(max_indexes) == 1)
           predicted_class = class_set(max_indexes);
           accuracy = class_set(max_indexes)== test_data(test_row, end);
        else
           if find(class_set(max_indexes) == test_data(test_row, end))
               predicted_class = datasample(class_set(max_indexes),1);
               accuracy = 1/length(max_indexes);  
           else
               predicted_class = datasample(class_set(max_indexes),1);
               accuracy = 0;
           end
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', test_row-1, predicted_class, test_data(test_row, end), accuracy);
        t_accu = [t_accu,accuracy];
    end
    fprintf('classification accuracy=%6.4f\n', sum(t_accu)/length(t_accu));
end

