function svd_power(data_file, dimen, iteration)

    data_set = importdata(data_file);
    dimen = str2num(dimen);
    iteration = str2num(iteration);
    
    u_mat = zeros(size(data_set,1), dimen);
    v_mat = zeros(size(data_set,2), dimen);
    s_mat = eye(dimen, dimen);
    
    for index = 1:dimen
        [u_mat(:,index), s_mat(index,index)] = pow_method(data_set*data_set', iteration);
        [v_mat(:,index),t] = pow_method(data_set'*data_set, iteration);
        data_set = data_set - (u_mat(:,index) * v_mat(:,index)')*s_mat(index,index);
    end
    
    recons_data = u_mat*s_mat*v_mat';
    display_mat('U', u_mat);
    display_mat('S', s_mat);
    display_mat('V', v_mat);
    display_mat('Reconstruction (U*S*V'')', recons_data);
end


function [vect, lamda] = pow_method(cov_mat, itr)
    
    b = ones(size(cov_mat,2),1);
    for r = 1:itr
        b = cov_mat*b;
        lamda = sqrt(norm(b));
        b = b/norm(b);
    end
    vect = b;
end

function display_mat(mname, mat_data)

    fprintf('Matrix %s:\n', mname);
    for i = 1:size(mat_data,1)
        fprintf(' Row %3d: %s\n', i, sprintf('%8.4f',mat_data(i,:)));
    end
    fprintf('\n');
    
end


