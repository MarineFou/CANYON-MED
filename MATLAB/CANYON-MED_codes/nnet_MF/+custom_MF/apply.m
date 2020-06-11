function a = apply(n,param)
%CUSTOM_MF.APPLY Apply transfer function to inputs

% Copyright 2018, MF.


% alpha=1.7159;
% A=4/3;
alpha=4/3;
A=1.7159;

a=A*((exp(alpha*n)-1)./(exp(alpha*n)+1));

end


