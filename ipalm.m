classdef ipalm < matlab.mixin.SetGet & matlab.mixin.Copyable
%IPALM  An iPALM iterator esp. suited to bilinear problems on the sphere
%
%   Iterates on problems based on models involving A, X, and b,
%     A:  dictionaries interacting with X on the oblique manifold 
%     X:  a map that combines atoms in the dictionary in some way
%     b:  biases which are otherwise not involved in the model
%
%   Examples include (convolutional) dictionary learning, sparse blind
%   deconvolution, analysis dictionary learning, and so forth:
%       e.g.    Y = A o X + b,      X = A'o Y + b.
%
%   The structure of the iterator stays true enough to this bilinear model
%   to keep its implementation simple whilst offering enough generality for
%   extensions. Just provide/adjust its various properties,
%
%     NONSEPERABLE TERM  H:  struct/object. 
%       Provide the following fields,
%
%     SEPERABLE TERMS  f:  cell array of structs/objects. 
%       Must have the same number of elements as X (see below). Each cell 
%       element is a struct/object with the fields
%
%     VARIABLES  A, X, b:  cell array. 
%       Each cell element contains a dictionary / combination map / bias 
%       -- e.g. in an array. A and X must have the same number of cells 
%       (since they interact in pairs). The number of cells in b is not 
%       restricted since depends on the data.
%
%     STEPSIZES  tA, tX, tb:  cell array of function handles.
%       tA must have the same number of cells as A, etc. Each cell should 
%       be a function handle of the form  
%           @ (cache, A, X, b) [DOUBLE > 0]
%       that returns the stepsize. Here CACHE is the iPALM cache.
%
%     MOMENTUM TERM  alph:  double in (0,1).
%

properties
    H;  f;              % nonseperable / seperable costs
    A;  X;  b;          % variables
    A_;  X_;  b_;       % momentum terms
    
    tA; tX; tb;         % stepsizes
    alph = 0.99;        % momentum parameter
    
    cost;
    it = 0;
    info = struct('debug', false);
end

properties (Access = private)
    A0;  X0;  b0;       % fixed initializations
    ob = obops;         % operations on the oblique manifold
    cache = struct;
end

methods
function obj = ipalm(H, f, A0, X0, b0, tA, tX, tb)
    %IPALM  Construct an iPALM iterator
    obj.H = H;
    obj.f = f;
    
    obj.A0 = A0;  obj.A = A0;  obj.A_ = A0;  
    obj.X0 = X0;  obj.X = X0;  obj.X_ = X0;
    obj.b0 = b0;  obj.b = b0;  obj.b_ = b0;
    
    obj.tA = tA;  obj.tX = tX;  obj.tb = tb;
end

function o = reset(o, A0, X0, b0)
    %RESET  Reset the iPALM iterator
    if nargin >= 2 || ~isempty(A0);  o.A0 = A0;  end
    if nargin >= 3 || ~isempty(X0);  o.X0 = X0;  end
    if nargin >= 4 || ~isempty(b0);  o.b0 = b0;  end
    
    o.A  = o.A0;  o.X  = o.X0;  o.b  = o.b0;
    o.A_ = o.A0;  o.X_ = o.X0;  o.b_ = o.b0;
    
    o.cost = [];
    o.it = 0;
    
    o.cache = struct;
end

function o = iterate(o)  %#ok<*NASGU>
    %ITERATE  Compute one iteration of iPALM
    Asz = size(o.A);  Xsz = size(o.X);  bsz = size(o.b); 
    
    assert(isequal(Asz(2), Xsz(1)), ['Number of atom-map ' ...
        'pairs -- size(A,2) and size(X,1) -- must be the same']);
    
    assert(isequal(numel(o.X), numel(o.f)), ['Number of maps and ' ...
        'regularizers -- numel(X) and numel(f) -- must be the same']);
    
    assert((numel(o.A) == numel(o.tA)) && ...
        (numel(o.X) == numel(o.tX)) && (numel(o.b) == numel(o.tb)), ...
        ['Number of stepsize elements -- tA, tX and tb -- must be the ' ...
        'same as A, X and b, respectively.']);
    
    c = o.cache;
    if o.info.debug
        o.info.gradA = cell(Asz);  o.info.tA = cell(Asz);
        o.info.gradX = cell(Xsz);  o.info.tX = cell(Xsz);
        o.info.gradb = cell(bsz);  o.info.tb = cell(bsz);
    end
    
    for i = 1:size(o.X,1)
        % Prox steps with momentum over X(i,:)
        for j = 1:size(o.X, 2)
            w = o.X;  
            w{i,j} = w{i,j} + o.alph*(w{i,j} - o.X_{i,j});
            [g, c] = o.H.gradX{i,j}(o.A, w, o.b, c);
            [t, c] = o.tX{i,j}(o.A, w, o.b, c);

            o.X_{i,j} = o.X{i,j};
            o.X{i,j} = o.f{i,j}.prox(w{i,j} - t*g, 1/t);
            
            if o.info.debug
                o.info.gradX{i,j} = g;
                o.info.tX{i,j} = t;
            end
        end
        
        % Riemmanian steps with momentum over A(:,i)
        for j = 1:size(o.A, 1)
            w = o.A;  
            w{j,i} = o.ob.Exp(w{j,i}, o.alph*o.ob.Log(o.A_{j,i}, w{j,i}));
            [g, c] = o.H.gradA{j,i}(w, o.X, o.b, c);
            [t, c] = o.tA{j,i}(w, o.X, o.b, c);

            o.A_{j,i} = o.A{j,i};
            o.A{j,i} = o.ob.Exp(w{j,i}, -t*g);
            
            if o.info.debug
                o.info.gradA{j,i} = g;
                o.info.tA{j,i} = t;
            end
        end
    end
    
    for i = 1:numel(o.b)
        % Momentum steps over b
        w = o.b;  w{i} = w{i} + o.alph*(w{i} - o.b_{i});
        
        [g, c] = o.H.gradb{i}(o.A, o.X, w, c);
        [t, c] = o.tb{i}(o.A, o.X, w, c);
        
        o.b_{i} = o.b{i};
        o.b{i} = w{i} - t*g;
        
        if o.info.debug
            o.info.gradb{i} = g;
            o.info.tb{i} = t;
        end
    end
    
    % Compute the cost and increment iteration count
    [o.cost, c] = o.H.value(o.A, o.X, o.b, c);
    for i = 1:numel(o.X)
        o.cost = o.cost + o.f{i}.value(o.X{i});
    end
    
    o.cache = c;  
    if o.info.debug;  o.info.cache = c;  end
    o.it = o.it + 1;
end
end
end

