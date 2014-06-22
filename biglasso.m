function [W, W_dual, Psi, Theta] = biglasso(S, T, lambda, varargin)

% BIGLASSO Sparse-inverse covariance estimation with the bigraphical lasso.
%
% FORMAT
% DESC Estimates a sparse precision (inverse-covariance) matrix through a
% series of LASSO regressions on partitions of the sample covariance
% matrix. The main difference from the glasso algorithm is the
% parametrisation of the precision matrix as a Kronecker sum of two sparse
% precision matrices, Psi and Theta. This is also known in algebraic graph
% theory as the Cartesian product of two graphs (since the precisions
% individually inscribe connectivities of a GMRF).
% The Kronecker sum of Psi and Theta is defined as:
%
%       kron(Psi, eye(size(Theta))) + kron(eye(size(Psi)), Theta).
%
% From this, is it clear that the Kronecker sum is decomposed into the
% precisions involved in a product of Gaussian distributions. As such,
% the bigraphical lasso can learn a model for any design matrix with
% dependencies across columns (Theta) as well as across rows (Psi).
%
% ARG S : the sample covariance matrix (the outer product matrix X'X/n). We
% assume S comes from data with standardised predictors (columns with zero
% means, unit norms). It has size d x d.
%
% ARG T : the feature covariance matrix (the inner product matrix XX'/d).
% we assume T comes from data with standardised datapoints (rows with zero
% means, unit norms). It has size n x n.
%
% ARG lambda : the L1 shrinkage (regularisation) parameter.
%
% The function accepts a variable length input argument list, in the format
%   ['arg1name', arg1value , 'arg2name', arg2value, ...].
% Available options are:
%
%   'warnInit':     Initialisation for W (estimated covariance).
%   'maxIter' :     Maximum number of iterations.
%   'thresh'  :     Convergence threshold for coordinate descent. Each
%                   inner coordinate-descent loop continues until the
%                   relative change in any coefficient is less than
%                   thresh. Defaults value is 1E-4.
%
% RETURN W : the estimated covariance matrix, inv(kronSum(Psi, Theta)).
%
% RETURN W_dual : the estimated dual covariance matrix, inv(kronSum(Theta, Psi)).
%
% RETURN Psi : the estimated row-precision matrix.
%
% RETURN Theta : the estimated column-precision matrix.
%
% SEEALSO : glmnet, glasso
%
% COPYRIGHT : Alfredo Kalaitzis, 2012 - 2014
%
% BIGLASSO

% addpath(genpath('./glmnet_matlab/'))                                        % Friedman's LASSO.
% addpath(genpath('./L1General/'))                                            % Schmidt's L1GeneralProjection.

if nargin < 3
    error('The feature covariance, datapoint covariance and regularisation parameter must be given.')
else
    if all(lambda < 0)
        error('lambda must be non-negative.')
    end
    p = size(S, 2);     n = size(T, 2);
    [~, pd] = chol(S + lambda(1)*eye(p));   % ISSUE: depending on the relative size of n and p, either S or T will be low-rank.
    if pd > 0
        error('S is not a proper covariance.')
    end
    [~, pd] = chol(T + lambda(2)*eye(n));
    if pd > 0
        error('T is not a proper covariance.')
    end
end

for k = 1:2:length(varargin)
    switch varargin{k}
        case 'warmInit'; warmInit = varargin{k+1};
            if ~isempty(warmInit)
                W_new = warmInit{1};    W_dual_new = warmInit{2};
                Theta = warmInit{3};    Psi = warmInit{4};
            end
        case 'maxIter'; maxIter = varargin{k+1};
        case 'thresh'; thresh = varargin{k+1};
    end
end

if ~exist('warmInit','var') || isempty(warmInit)
    W_new = kron(T,S) + prod(lambda)*eye(n*p);                                    % W is the estimated sparse-inverse. Default init. as kronecker product of sample covariances + lambda*I.
    W_dual_new = kron(S,T) + prod(lambda)*eye(n*p);
%     [~, Theta] = glasso(S, lambda, 'maxIter', maxIter, 'thresh', thresh);   % Initialise precisions with independent runs of graphical lasso. Could also use identity.
%     [~, Psi] = glasso(T, lambda, 'maxIter', maxIter, 'thresh', thresh);
    Theta = eye(p);     Psi = eye(n);    
end
if ~exist('maxIter','var')
    maxIter = 100;                                                          % Maximum number of glasso iterations.
end
if ~exist('thresh','var')
    thresh = 1e-4;                                                          % Convergence threshold on W (estimated inverse-covariance).
end
options = glmnetSet;                                                        % glmnet options.
options.L1GPoptions.verbose = 0;

converged = false;
iter = 1;
W = W_new;  W_dual = W_dual_new;

while ~converged && (iter < maxIter)
    options.lambda = lambda(2);                                                    % Regularization parameter.
    [Psi, W_new] = sub_biglasso(W_new, T, Psi, Theta, options);                 % Arguments in sequence of passing: current_covariance_estimate, sample_covariance, current_precision_estimate, fixed_parameter.
    Psi = Psi ./ Psi(1,1);                                                          % Normalise Psi s.t. psi_11 = 1.
    options.lambda = lambda(1);                                                    % Regularization parameter.
    [Theta, W_dual_new] = sub_biglasso(W_dual_new, S, Theta, Psi, options);     % First optimise Psi with Theta fixed, then Theta with Psi fixed.
    Theta = Theta ./ Theta(1,1);                                                    % Normalise Psi s.t. psi_11 = 1.    
    
    diffW = abs(W - W_new);     diffW_dual = abs(W_dual - W_dual_new);          % Check for convergence of W.
    if all((diffW(:) + diffW_dual(:)) < thresh)
        converged = true;
    end
    
    iter = iter + 1;
    fprintf('iter: %d diff: %1.4f \n\n', iter, sum(diffW(:)+diffW_dual(:)))
    W = W_new;      W_dual = W_dual_new;
end

end
%------------------------------------------------------------------
% End of function biglasso
%------------------------------------------------------------------


function [Psi, W_new] = sub_biglasso(W_new, T, Psi, Theta, options)         % Main subroutine: Provides updates of the optimised parameter.
p = size(Theta, 2);     n = size(Psi, 2);
spIp = speye(p);
options.order = -1;         % -1: L-BFGS (limited-memory), 1: BFGS (full-memory), 2: Newton
options.verbose = 0;

%fprintf('Vars done: ');
for i = 1:n
    rIndx_11 = [1 : (i-1)*p     i*p + 1 : n*p];
    rIndx_t12 = [1:(i-1)   (i+1):n];
    W11 = W_new( rIndx_11, rIndx_11 );                                      % Partition W. Blocks are p x p. This is the large chunk: all but the ith block-row and block-column of W.
    t12 = T( rIndx_t12, i );                                                % ith column of T (datapoint sample covariance) w/o ith row.
    
    % Theta_inv = sparse(U * diag( 1 / (diag(D)+Psi(i,i)) ) * U');                    % Fast update of inverse, using the eigen-decomposition of Theta.
    Theta_inv = inv(Psi(i,i)*eye(p) + Theta) ;
    Theta_plus = Psi(i,i)*eye(p) + Theta;
    % blkD = kron( spIn_1, Theta_inv );
    A = (1/p) * blkwiseTraceProduct(W11, Theta_inv, p);
    
    fit = glmnet(A, -t12, 'gaussian', options);                             % Lasso regression of the ith component on the rest, using pathwise coordinate descent.
    %funObj   = @(x)GaussianLoss(x, A, -t12);                                % Loss function that L1 regularization is applied to
    %fit.beta = L1GeneralProjection(funObj, zeros(p-1, 1), options.lambda*ones(p-1,1), options);
    
    psi12 = sparse(fit.beta);
    Psi( rIndx_t12, i ) = psi12;
    Psi( i, rIndx_t12 ) = psi12';
    
    omega12 = kron( psi12, spIp );                                          % Omega is the big Kronecker-sum precision (inverse of W).
    w12_new = - W11 * (omega12 / Theta_plus);                                % Update ith block-column of sparse-inverse W.
    % w12_new = - W11 * kron(psi12, Theta_inv);
    w22 = (eye(p) - w12_new' * omega12) / Theta_plus;
    
    cIndx_12 = (i-1)*p + 1 : i*p;
    W_new( rIndx_11, cIndx_12 ) = w12_new;
    W_new( cIndx_12, rIndx_11 ) = w12_new';
    W_new( cIndx_12, cIndx_12 ) = w22;
    fprintf('%d ', i);
end
fprintf('\n');
end
%------------------------------------------------------------------

function bwt = blkwiseTraceProduct(A, B, s)                                 % Returns a matrix of traces of s x s blocks in A * blkdiag(B).
numBlocks = size(A, 2) / s;                                                 %   We assume B is a repeated block in a block-diagonal,
bwt = zeros(numBlocks);                                                     %   with block size s x s. This allows for efficient computation of bwt.
rIndx = 1:s;
for i = 1:numBlocks
    cIndx = 1:s;
    for j = 1:numBlocks
        % bwt(i,j) = traceProduct( A(rIndx, cIndx), B(cIndx, cIndx) );
        bwt(i,j) = traceProduct( A(rIndx, cIndx), B );
        cIndx = cIndx + s;
    end
    rIndx = rIndx + s;
end
end
%------------------------------------------------------------------

function t = traceProduct(A, B)                                             % Efficiently computes the trace of a matrix product.
t = sum(sum(A.*B'));
end
%------------------------------------------------------------------

function bdt = blkdiagTrace(A, s)                                           % Returns the sum along the block diagonal.
bdt = zeros(s);
numBlocks = size(A, 2) / s;
for i = 1:numBlocks
    rIndx = (i-1)*s + 1 : i*s;
    bdt = bdt + A(rIndx, rIndx);
end
end
