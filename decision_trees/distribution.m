function [ dist ] = distribution( samples, class_set )
    t_size = size(samples,1);
    dist = [];
    for class_in = 1:length(class_set)
       dist = [dist,size(samples(samples() == class_set(class_in)),1)/t_size];
       
    end
end

