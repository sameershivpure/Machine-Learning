function [ tree ] = dtree_learning(samples, attr, deflt, pr_th, class_set, type)
    
    if size(samples,1) < pr_th
        tree = deflt;
    elseif length(unique(samples(:,end))) == 1
        tree = distribution(samples(:,end), class_set);
    else
        tree = [];
        [best_attr, best_th, m_gain] = choose_attribute(samples, attr,class_set, type);
        tree.('attr') = best_attr;
        tree.('thres') = best_th;
        tree.('gain') = m_gain;
        left_set = samples(samples(:,best_attr) < best_th, :);
        right_set = samples(samples(:,best_attr) >= best_th, :);
        tree.('left_child') = dtree_learning(left_set, attr, distribution(samples(:,end),class_set), pr_th, class_set, type);
        tree.('right_child') = dtree_learning(right_set, attr, distribution(samples(:,end), class_set), pr_th, class_set, type);
        
    end
end

