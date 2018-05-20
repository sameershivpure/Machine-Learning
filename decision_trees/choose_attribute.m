function [ best_attribute, best_threshold, max_gain ] = choose_attribute( samples, attr_list, class_set, type )
    
    max_gain = -1;
    best_attribute = -1;
    best_threshold = -1;
    
    if type == 1
        for attr_in = 1: length(attr_list)
           attr_vals = samples(:,attr_in);
           min_val = min(attr_vals);
           max_val = max(attr_vals);
           for th_in = 1:50
              th = min_val + th_in*(max_val-min_val)/51;
              gain = information_gain(samples, attr_in, th, class_set);
              if gain > max_gain
                 max_gain = gain; 
                 best_attribute = attr_in;
                 best_threshold = th;
              end
           end
        end
    elseif type == 0
       attr_in = datasample(attr_list,1,'Replace',false);
       attr_vals = samples(:,attr_in);
       min_val = min(attr_vals);
       max_val = max(attr_vals);
       for th_in = 1:50
          th = min_val + th_in*(max_val-min_val)/51;
          gain = information_gain(samples, attr_in, th, class_set);
          if gain > max_gain
             max_gain = gain; 
             best_attribute = attr_in;
             best_threshold = th;
          end
       end
    end
end

function [gain] = information_gain (samples, attr, thres, class_set)

    t_size = size(samples,1);
    ent = 0;
    lent = 0;
    rent = 0;
    left = samples(samples(:,attr) < thres,:);
    right = samples(samples(:,attr) >= thres,:);
    
    for class_in = 1:length(class_set)
       pr = size(samples(samples(:,end)== class_set(class_in)),1)/t_size;
       if pr > 0
           ent = ent + (-pr*log2(pr));
       end
       
       lpr = size(left(left(:,end)== class_set(class_in)),1)/size(left,1);
       if lpr > 0
          lent = lent + (-lpr*log2(lpr));
       end
       
       rpr = size(right(right(:,end) == class_set(class_in)),1)/size(right,1);
       if rpr > 0
          rent = rent + (-rpr*log2(rpr)); 
       end
    end
    
    gain = ent - (size(left,1)/t_size)*lent - (size(right,1)/t_size)*rent;
    
end