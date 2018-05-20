function naive_bayes(training_file, test_file, classify_type, parts)

training_data = importdata(training_file);
test_data = importdata(test_file);
if nargin < 4
    parts = '1';
end
training_data = sortrows(training_data, size(training_data,2));
classes = unique(training_data(:,end));
dimensions = size(training_data,2)-1;
prob_of_class = zeros(length(classes),1);

test_targets = test_data(:,end);
test_data = test_data(:,1:end-1);
predicted_class = [];
predicted_prob = [];
accuracy = [];


if strcmpi(classify_type,'histograms')
   priors = zeros(dimensions, length(classes), str2num(parts));  
   bin_interval = zeros(dimensions, length(classes), str2num(parts)+1);
   for class_index = 0:length(classes)-1
      class_sample = training_data(training_data(:,end)==classes(class_index+1),1:(end-1));
      prob_of_class(class_index+1) = size(class_sample, 1)/size(training_data, 1);
      max_value = max(class_sample);
      min_value = min(class_sample);
      if str2num(parts) > 3
        gfactor = (max_value - min_value)/(str2num(parts)-3);
      else
        gfactor = (max_value - min_value)/(str2num(parts));
      end
      gfactor(gfactor() < 0.0001) = 0.0001;
      bin_interval(:,class_index+1,1) = -inf;
      bin_interval(:,class_index+1,2) = (min_value - gfactor/2)';
      bin_interval(:,class_index+1,end) = inf;
      for bin_count  = 0:str2num(parts)-3
        bin_interval(:,class_index+1,bin_count+3) = (min_value + (bin_count)*gfactor + gfactor/2)';
      end
      for dimen = 0:dimensions-1
          for bin_index = 0:size(bin_interval,3)-2
          priors(dimen+1,class_index+1,bin_index+1) =  size(class_sample((class_sample(:,dimen+1)>= bin_interval(dimen+1,class_index+1,bin_index+1)& class_sample(:,dimen+1)<bin_interval(dimen+1, class_index+1, bin_index+2)),dimen+1),1);
          end
      end
      priors(:,class_index+1,:) = priors(:,class_index+1,:)./(size(class_sample,1)*gfactor)';
   end
   for class_in = 0:length(classes)-1
        for dimen_in = 0:dimensions-1
            for bin_in = 0:str2num(parts)-1
                fprintf('class %d, attribute %d, bin %d, P(bin | class) = %.2f\n', classes(class_in+1), dimen_in, bin_in, priors(dimen_in+1, class_in+1, bin_in+1));
        
            end
        end
   end
   
   
   for test_row = 0:size(test_data,1)-1
       prob_data_given_class = zeros(length(classes),1);
       for test_class_in = 0:length(classes)-1
           prob_dimension_given_class = 1;
           for test_dimen = 0:dimensions-1
               pred_bin = 0;
               for test_bin = 0:size(bin_interval,3)-2
                    if (bin_interval(test_dimen+1, test_class_in+1, test_bin+1) <= test_data(test_row+1,test_dimen+1) & test_data(test_row+1,test_dimen+1) < bin_interval(test_dimen+1, test_class_in+1, test_bin+2))
                        pred_bin = test_bin+1;
                        break;
                    end
               end
               prob_dimension_given_class = prob_dimension_given_class*priors(test_dimen+1, test_class_in+1, pred_bin);
               
           end
           prob_data_given_class(test_class_in+1) = prob_dimension_given_class * prob_of_class(test_class_in+1);
       end
       if sum(prob_data_given_class) > 0
        prob_data_given_class = prob_data_given_class/sum(prob_data_given_class);
       end
       max_prob = max(prob_data_given_class);
       max_indexes = find(prob_data_given_class == max_prob);
       if (size(max_indexes,1) == 1)
           predicted_class = [predicted_class; classes(max_indexes)];
           predicted_prob = [predicted_prob; max_prob];
           accuracy = [accuracy; classes(max_indexes)== test_targets(test_row+1)];
       else
           if find(classes(max_indexes) == test_targets(test_row+1))
               predicted_class = [predicted_class; test_targets(test_row+1)];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 1/size(max_indexes,1)];  
           else
               predicted_class = [predicted_class; classes(max_indexes(1))];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 0];
           end
       end
   end
   
   fprintf('\n Test data classification result\n')
   for test_in = 0:size(test_data,1)-1
      fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f \n', test_in, predicted_class(test_in+1), predicted_prob(test_in+1), test_targets(test_in+1), accuracy(test_in+1)) 
   end
   
   fprintf('classification accuracy = %6.4f\n',sum(accuracy)/size(accuracy,1))
   
   
elseif strcmpi(classify_type,'gaussians')
    class_mean = [];
    class_std = [];
    for class_index = 0:length(classes)-1
      class_sample = training_data(training_data(:,end)==classes(class_index+1),1:(end-1));
      prob_of_class(class_index+1) = size(class_sample, 1)/size(training_data, 1);
      class_mean = [class_mean; mean(class_sample)];
      class_std = [class_std; std(class_sample)];
    end
    class_std(class_std() < 0.01) = 0.01;
    for class_in = 0:length(classes)-1
        for dimen_in = 0:dimensions-1
                fprintf('class %d, attribute %d, mean = %.2f, std = %.2f\n', classes(class_in+1), dimen_in, class_mean(class_in+1, dimen_in+1), class_std(class_in+1, dimen_in+1));
        end
    end
    
    for test_row = 0:size(test_data,1)-1
       prob_data_given_class = zeros(length(classes),1);
       for test_class_in = 0:length(classes)-1
           
            prob_data_given_class(test_class_in+1) = prod(normpdf(test_data(test_row+1,:), class_mean(test_class_in+1,:), class_std(test_class_in+1,:)))* prob_of_class(test_class_in+1);
       end
       prob_data_given_class = prob_data_given_class/sum(prob_data_given_class);
       max_prob = max(prob_data_given_class);
       max_indexes = find(prob_data_given_class == max_prob);
       if (size(max_indexes,1) == 1)
           predicted_class = [predicted_class; classes(max_indexes)];
           predicted_prob = [predicted_prob; max_prob];
           accuracy = [accuracy; classes(max_indexes)== test_targets(test_row+1)];
       else
           if find(classes(max_indexes) == test_targets(test_row+1))
               predicted_class = [predicted_class; test_targets(test_row+1)];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 1/size(max_indexes,1)];  
           else
               predicted_class = [predicted_class; classes(max_indexes(1))];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 0];
           end
       end
    end
    
    fprintf('\n Test data classification result\n')
   for test_in = 0:size(test_data,1)-1
      fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f \n', test_in, predicted_class(test_in+1), predicted_prob(test_in+1), test_targets(test_in+1), accuracy(test_in+1)) 
   end
   
   fprintf('classification accuracy = %6.4f\n',sum(accuracy)/size(accuracy,1))
    
elseif strcmpi(classify_type,'mixtures')
   mixtures = str2num(parts);
   std_mixture = repmat(1, dimensions, length(classes), mixtures);
   weights = repmat(1/mixtures, dimensions, length(classes), mixtures);
   mean_mixture = repmat(0, dimensions, length(classes), mixtures);
   
   for class_index = 0:length(classes) -1
       class_sample = training_data(training_data(:,end)==classes(class_index+1),1:(end-1));
       prob_of_class(class_index+1) = size(class_sample, 1)/size(training_data, 1);
       max_value = max(class_sample);
       min_value = min(class_sample);
       gfactor = (max_value - min_value)/mixtures;
       for mixture_index = 0:mixtures-1
            mean_mixture(:,class_index+1,mixture_index+1) = (min_value + mixture_index*gfactor + gfactor/2)';
       end
   end
   
   for count = 0:49
       for class_index = 0:length(classes)-1
          class_sample = training_data(training_data(:,end)==classes(class_index+1),1:(end-1));
          for dimen = 0:dimensions-1
              prob = [];
              for mix_index = 0:mixtures-1
                    prob = [prob;(weights(dimen+1, class_index+1, mix_index+1)*normpdf(class_sample(:,dimen+1), mean_mixture(dimen+1, class_index+1, mix_index+1), std_mixture(dimen+1, class_index+1, mix_index+1)))'];
              end
              
              if sum(prob) > 0
                prob = prob./sum(prob);
              end
              
              for mix  = 0:mixtures-1
                    if sum(prob(mix+1,:)) > 0 
                        prob_sum = sum(prob(mix+1,:));
                    else
                        prob_sum = 1;
                    end

                    mean_mixture(dimen+1, class_index+1, mix+1) = prob(mix+1,:)*class_sample(:,dimen+1)/prob_sum;
                    std_mixture(dimen+1, class_index+1, mix+1) = sqrt((prob(mix+1,:)*(class_sample(:, dimen+1)- mean_mixture(dimen+1, class_index+1, mix+1)).^2)/prob_sum);
                    weights(dimen+1, class_index+1, mix+1) = sum(prob(mix+1,:))/sum(sum(prob));

              end

          end
       end
       std_mixture(std_mixture() < 0.01) = 0.01;
   end

   
   for class_in = 0:length(classes)-1
        for dimen_in = 0:dimensions-1
            for mix_in = 0:mixtures-1
                fprintf('class %d, attribute %d, Gaussian %d, mean = %.2f, std = %.2f\n', classes(class_in+1), dimen_in, mix_in, mean_mixture(dimen_in+1, class_in+1, mix_in+1), std_mixture(dimen_in+1, class_in+1, mix_in+1));
        
            end
        end
   end
   
   for test_row = 0: size(test_data,1)-1
       prob_data_given_class = zeros(length(classes),1);
       for test_class_in = 0:length(classes)-1
            test_mix_prob = [];
            for test_mix  = 0:mixtures-1   
                test_mix_prob = [test_mix_prob;(normpdf(test_data(test_row+1,:), mean_mixture(:,test_class_in+1,test_mix+1)', std_mixture(:,test_class_in+1,test_mix+1)')).* weights(:,test_class_in+1,test_mix+1)'];
            end
            prob_data_given_class(test_class_in+1) = prod(sum(test_mix_prob))* prob_of_class(test_class_in+1);
            
       end
       prob_data_given_class = prob_data_given_class/sum(prob_data_given_class);
       max_prob = max(prob_data_given_class);
       max_indexes = find(prob_data_given_class == max_prob);
       if (size(max_indexes,1) == 1)
           predicted_class = [predicted_class; classes(max_indexes)];
           predicted_prob = [predicted_prob; max_prob];
           accuracy = [accuracy; classes(max_indexes)== test_targets(test_row+1)];
       else
           if find(classes(max_indexes) == test_targets(test_row+1))
               predicted_class = [predicted_class; test_targets(test_row+1)];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 1/size(max_indexes,1)];  
           else
               predicted_class = [predicted_class; classes(max_indexes(1))];
               predicted_prob = [predicted_prob; max_prob];
               accuracy = [accuracy; 0];
           end
       end
   end
   
   fprintf('\n Test data classification result\n')
   for test_in = 0:size(test_data,1)-1
      fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f \n', test_in, predicted_class(test_in+1), predicted_prob(test_in+1), test_targets(test_in+1), accuracy(test_in+1)) 
   end
   
   fprintf('classification accuracy = %6.4f\n',sum(accuracy)/size(accuracy,1))
    
end




