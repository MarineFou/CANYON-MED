function dn = backprop(da,n,a,param)
%LOGSIG.BACKPROP Backpropagate derivatives from outputs to inputs

% Copyright 2012-2015 The MathWorks, Inc.

  dn = bsxfun(@times,da,1-(a.*a));
end
