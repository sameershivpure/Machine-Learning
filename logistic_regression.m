function logistic_regression(train_file, poly_deg, test_file)

train_set = importdata(train_file);
train_targets = train_set(:,end);
train_set = train_set(:,1:end-1);
train_targets(train_targets() ~= 1) = 0;

test_set = importdata(test_file);
test_targets = test_set(:,end);
test_set = test_set(:,1:end-1);
test_targets(test_targets() ~= 1) = 0;

poly_deg = str2num(poly_deg);
dimensions = size(train_set, 2);
weights = zeros(poly_deg*dimensions+1,1);
phi_x = zeros(size(train_set,1), (poly_deg*dimensions)+1);
phi_x(:,1) = 1;
done = 0;

for row_index = 1:size(train_set,1)
    dim_in = 2;
    for dimen_index = 1:dimensions
        for deg = 1:poly_deg
            phi_x(row_index,dim_in) = train_set(row_index, dimen_index).^deg;
            dim_in = dim_in +1;
        end
    end  
end

old_error = 0;
while(~done)
    y_wx = ((1+ exp(-(phi_x*weights))).^(-1));
    new_error = sum(abs(train_targets.*log(y_wx) + (1 - train_targets).*log(1-y_wx)));
    if abs(new_error-old_error) < 0.001
       done = 1;
       break 
    end
    diag_R = diag(y_wx.*(1-y_wx));
    grad = inv(phi_x.'*diag_R*phi_x)*phi_x.'*((y_wx) - train_targets);
    weights = weights - grad;
    
    if sum(abs(grad)) < 0.001
        done = 1;
        break
    end
    old_error = sum(abs(train_targets.*log(y_wx) + (1 - train_targets).*log(1-y_wx)));
end
for w_in = 0:size(weights,1)-1
    fprintf('w%d=%.4f\n', w_in,weights(w_in+1,1));
end

% testing procedure
test_phi_x = zeros(size(test_set,1), (poly_deg*dimensions)+1);
test_phi_x(:,1) = 1;
for test_row_index = 1:size(test_set,1)
    dim_in = 2;
    for test_dimen_index = 1:dimensions
        for deg = 1:poly_deg
            test_phi_x(test_row_index,dim_in) = test_set(test_row_index, test_dimen_index).^deg;
            dim_in = dim_in +1;
        end
    end  
end
test_y = ((1+ exp(-(test_phi_x*weights))).^(-1));
predicted_class = round(test_y);
accuracy = 1 - abs(predicted_class - test_targets);
accuracy(test_y() == 0.5) = 0.5;

fprintf('\n Classification result\n');
for row = 0:size(test_set,1)-1
    if predicted_class(row+1,1) == 1
        prob = test_y(row+1,1);
    else
        prob = 1 - test_y(row+1,1);
    end
    fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',row, predicted_class(row+1,1), prob, test_targets(row+1,1), accuracy(row+1,1));
end

fprintf('classification accuracy=%6.4f\n', sum(accuracy)/size(accuracy,1));
end

