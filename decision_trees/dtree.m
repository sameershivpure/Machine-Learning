function dtree(train_file, test_file, option, prun_thr)

    train_set = importdata(train_file);
    test_set = importdata(test_file);
    prun_thr = str2num(prun_thr);
  
    classes = unique(train_set(:,end));
    attributes = 1:1:size(train_set,2)-1;
    dist = distribution(train_set(:,end), classes);
    
    if strcmpi(option,'optimized')
       dtree = dtree_learning(train_set,attributes,dist, prun_thr, classes, 1);
       display_tree(dtree,0);
       dtree_classify(test_set, classes, [dtree]);
    elseif strcmpi(option,'randomized')
       dtree = dtree_learning(train_set,attributes,dist, prun_thr, classes, 0);
       display_tree(dtree,0);
       dtree_classify(test_set, classes, [dtree]);
    elseif strcmpi(option,'forest3')
       dforest = [];
       for tin = 1:3
           dtree = dtree_learning(train_set,attributes,dist, prun_thr, classes, 0);
           dforest = [dforest, dtree];
           display_tree(dtree,tin-1);
       end
       dtree_classify(test_set, classes, dforest);
    elseif strcmpi(option,'forest15')
       dforest = [];
       for tin = 1:15
           dtree = dtree_learning(train_set,attributes,dist, prun_thr, classes, 0);
           dforest = [dforest, dtree];
           display_tree(dtree,tin-1);
       end
       dtree_classify(test_set, classes, dforest);
    end
end

function display_tree(dt,tid)
    queue = [dt];
    index = 1;
    q = 1;
    while q == 1
        
        fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n', tid, index, (queue(index).('attr')-1), queue(index).('thres'), queue(index).('gain'));
        if queue(index).('attr') ~= 0
            if strcmpi(class(queue(index).('left_child')),'struct')
                queue = [queue,queue(index).('left_child')];
            else
                node.('attr') = 0;
                node.('thres') = -1;
                node.('gain') = 0;
                node.('left_child') = 0;
                node.('right_child') = 0;
                queue = [queue,node];
            end
            if strcmpi(class(queue(index).('right_child')),'struct')
                queue = [queue,queue(index).('right_child')];
            else
                node.('attr') = 0;
                node.('thres') = -1;
                node.('gain') = 0;
                node.('left_child') = 0;
                node.('right_child') = 0;
                queue = [queue,node];
            end
        end
        ql = length(queue);
        if index == ql
            q= 0;
        end
        index  = index+1;
    end
end
