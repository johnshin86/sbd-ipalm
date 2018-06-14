classdef sbd_ipalm < ipalm
    
methods
function o = sbd_ipalm(Y, p, lambda, xpos, getbias)

    m = size(Y{1});  ob = obops; N = numel(Y);
    afun = @(funh, a) arrayfun(funh, a{:}, 'UniformOutput', false);
    H.value = @(A, X, b, c) Hval(A, X{:}, b, Y, c);

    % Set up A
    A0 = cell(N,1);
    for n =1:N
        tmp = [randi(m(1)) randi(m(2))];
        tmp = {mod(tmp(1)+(1:p(1)), m(1))+1 mod(tmp(2)+(1:p(2)), m(1))+1};
        A0{n}(:,:) = ob.proj(Y{n}(tmp{1}, tmp{2}));
    end
    
    H.gradA = afun(@(i) @(A, X, b, c) gradA(A, X{:}, b, Y, i, c), {(1:N)'});
    tA = afun(@(i) @(A, X, b, c) stepszA(A, X{:}, b, i, c), {(1:N)'});

    % Set up X
    H.gradX = {@(A, X, b, c) gradX(A, X{:}, b, Y, c)};
    tX = {@stepszX};
    f = {huber(lambda, xpos)};
    X0 = {ones(m)};

    % Set up b
    H.gradb = afun(@(i) @(A, X, b, c) gradb(A, X{:}, b, Y, i, getbias, c), {1:N});
    tb = afun(@(i) @(A,~,~,c) stepszb(A,[],[], Y, i, getbias, c), {1:N});
    if getbias
        b0 = afun(@(i) median(Y{i}(:)), {1:N});
    else
        b0 = afun(@(i) 0, {1:N});
    end
    
    o = o@ipalm(H, f, A0, X0, b0, tA, tX, tb);
end

end
end

function [v, cache] = Hval(A, X, b, Y, cache)
    if nargin < 5 || isempty(cache)
        N = numel(Y);
        v=0;
        for n = 1:N
            v = v + norm(cconvfft2(A{n}, X) + b{n} - Y{n}, 'fro')^2/2;
        end
        
    else
        v = sum(cache.Hcost);
    end
end

function [g, cache] = gradA(A, X, b, Y, n, cache)
    N = numel(Y);
    LAn=0;
    Xhat = fft2(X);  
    g = ifft2(conj(Xhat) .* (fft2(A{n},size(Y{n},1),size(Y{n},2)) .* Xhat + fft2(b{n}-Y{n})));
    g = real(g(1:size(A{n},1), 1:size(A{n},2)));
    LAn = max(LAn, max(abs(Xhat(:))));
    if nargin >= 6 && isstruct(cache)
        if ~isfield(cache, 'LA')
            cache.LA = zeros(size(A));  
        end
        cache.LA(n) = sqrt(N)*LAn^2;
    end
end

function [tAk, cache] = stepszA(~, ~, ~, n, cache)
tAk = 0.99/cache.LA(n);
end

function [g, cache] = gradX(A, X, b, Y, cache)
    Xhat = fft2(X); N = numel(Y); Ahat = cell(1,N); g=zeros(size(X));
    for n=1:N
        Ahat{n} = fft2(A{n}, size(Y{n},1), size(Y{n},2));
        g = g + real(ifft2(conj(Ahat{n}) .* (Ahat{n} .* Xhat + fft2(b{n}-Y{n}))));
    end
    if nargin >= 5 && ~isempty(cache)
        cache_iter = 0;
        for n = 1:N
            cache_iter = max(abs(Ahat{n}(:)))^2 + cache_iter;
        end
        cache.tX = cache_iter;
    end
end

function [t, cache] = stepszX(~, ~, ~, cache)
t = 0.99/cache.tX;
end

function [g, cache] = gradb(A, X, b, Y, n, getbias, cache)
    R = cconvfft2(A{n}, X) + b{n} - Y{n};
    if getbias; g = sum(R(:));
    else; g = 0;
    end
    if nargin >= 6 && ~isempty(cache)
        if ~isfield(cache, 'Hcost');  cache.Hcost = zeros(numel(Y),1);  end
        cache.Hcost(n) = norm(R(:))^2/2;
    end
end

function [t, cache] = stepszb(~, ~, ~, Y, n, getbias, cache)
if getbias
    t = 1/(2 * numel(Y{n}));
else
    t = 0;  
end
end

