function neural_network(training_file, test_file, layers, units, rounds)
    
    train_data = importdata(training_file);
    test_data = importdata(test_file);
    train_classes = train_data(:,end);
    train_data = train_data(:,1:end-1);
    m = max(train_data(:));
    train_data = train_data/m;
    test_classes = test_data(:,end);
    test_data = test_data(:,1:end-1)/m;
    
    layers = str2num(layers);
    units = str2num(units);
    rounds = str2num(rounds);
    
    classes = unique(train_classes);
    total_class = length(classes);
    train_targets = zeros(size(train_data,1),total_class);
    weights = [];
    layer_pts = [];
    
    if layers == 2
        units = total_class;
    elseif layers < 2
        fprintf('Layer value should be atleast 2');
        return
    end
    
    for i = 1:size(train_targets,1)
        train_targets(i,find(classes == train_classes(i))) = 1;
    end
    
    for layer_index = 1:layers-1
        if layer_index == 1
            v = -0.05 + (0.05+0.05)*rand(size(train_data,2)+1,units);
        elseif layer_index == layers-1
            v = -0.05 + (0.05+0.05)*rand(units+1,total_class);
        else
            v = -0.05 + (0.05+0.05)*rand(units+1,units);
        end
        weights.(strcat('l',num2str(layer_index))) = v;
        layer_pts = [layer_pts,string(strcat('l',num2str(layer_index)))];
    end
  
    for r = 1:rounds
        for row_index = 1:size(train_data,1)
            outputs = [];
            for layer_index = 1:layers-1
                
                if layer_index == 1
                    inputs = [ones(1,1),train_data(row_index,:)];
                else
                    inputs = [ones(1,1), outputs.(char(layer_pts(layer_index-1)))];
                end
                outputs.(char(layer_pts(layer_index))) = logsig(inputs*weights.(char(layer_pts(layer_index))));
            end
            
            for trainlayer = layers-1:-1:1
               if trainlayer == layers-1
                   z = outputs.(char(layer_pts(trainlayer)));
                   grad = ( z - train_targets(row_index,:)).*z.*(1-z);
                   if trainlayer == 1
                       inp = [ones(1,1),train_data(row_index,:)];
                   else
                       inp = [ones(1,1),outputs.(char(layer_pts(trainlayer-1)))];
                   end
               else
                   z = [ones(1,1),outputs.(char(layer_pts(trainlayer)))];
                   grad = (grad*weights.(char(layer_pts(trainlayer+1))).').*z.*(1-z);
                   grad = grad(2:end);
                   if trainlayer == 1
                       inp = [ones(1,1),train_data(row_index,:)];
                   else
                       inp = [ones(1,1),outputs.(char(layer_pts(trainlayer-1)))];
                   end
               end
               weights.(char(layer_pts(trainlayer))) = weights.(char(layer_pts(trainlayer))) - ((0.98).^(r-1))*inp.'*grad;
            end
        end
    end
    
    fprintf('\nClassification result\n');
    total_accu = [];
    for test_row = 1:size(test_data,1)
        for layer_index = 1:layers-1

            if layer_index == 1
                x = [ones(1,1),test_data(test_row,:)];
            else
                x = [ones(1,1), z];
            end
            z = logsig(x*weights.(char(layer_pts(layer_index))));       
        end
        max_val = max(z);
        max_indexes = find(z == max_val);
        if (length(max_indexes) == 1)
           predicted_class = classes(max_indexes);
           accuracy = classes(max_indexes)== test_classes(test_row);
        else
           if find(classes(max_indexes) == test_classes(test_row))
               predicted_class = datasample(classes(max_indexes),1);
               accuracy = 1/length(max_indexes);  
           else
               predicted_class = datasample(classes(max_indexes),1);
               accuracy = 0;
           end
        end
    
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',test_row-1, predicted_class, test_classes(test_row), accuracy); 
        total_accu = [total_accu,accuracy];
    end
    fprintf('classification accuracy=%6.4f\n', sum(total_accu)/length(total_accu));
end
    
 