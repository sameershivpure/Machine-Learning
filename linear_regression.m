function linear_regression(training_file, deg, lamda)

if nargin < 2
    deg = '1';
    lamda = '0';
elseif nargin < 3
    lamda = '0';
end

train_data = importdata(training_file);
poly_deg = str2num(deg);
lamda = str2num(lamda);
train_inputs = train_data(:,1);
train_targets = train_data(:,2);

phi_x = zeros(size(train_inputs,1), poly_deg+1);
iden = eye(poly_deg+1);

for p_order = 0:poly_deg
    phi_x(:,p_order+1) = train_inputs.^p_order;
end
weights = (inv(lamda*iden + (phi_x.')*phi_x)*phi_x.')*train_targets;

fprintf('w0=%.4f\n', weights(1));
fprintf('w1=%.4f\n', weights(2));
if poly_deg > 1
    fprintf('w2=%.4f\n', weights(3));
else
    fprintf('w2=%.4f\n', 0);
end

end

