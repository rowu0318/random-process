function [decoded,numits] = decode_method(encoded,reference)

string=' ,-.0123456789;?abcdefghijklmnopqrstuvwxyz';
len = length(string);
P = zeros(len,len); % transition matrix

for i = 1:length(reference)-1
    row = strfind(string,reference(i));
    col = strfind(string,reference(i+1));
    P(row,col) = P(row,col) + 1;
end
P = P./sum(P,2) + (P==0)*1e-14;

% encoded to string map
encoded_len = length(encoded);
str = zeros(1,encoded_len);
decoded = encoded;
for j = 1:encoded_len
    str(j) = strfind(string,encoded(j));
end

% Initial random permutation
f = randperm(len);

logLf = 0; % initial decoding likelihood
for j = 1:encoded_len-1
    temp = P(f(str(j)),f(str(j+1)));
    logLf = logLf + log(temp);
end

numits = 0; % number of iteration
reject = 0;

% Permute until Lf maximum or no changes
while(reject < 50000)
    numits = numits + 1;
    
    % swap two characters in the decoding function
    fstar = f;
    swap = randsample(len,2);
    a = f(swap(1));
    b = f(swap(2));
    fstar(swap(1)) = b;
    fstar(swap(2)) = a;
    
    % Compute new log likelihoood
    logLf_star = 0;
    for j = 1:encoded_len-1
        temp = P(fstar(str(j)),fstar(str(j+1)));
        logLf_star = logLf_star + log(temp);
    end
    
    % Accept with Metropolis probability
    prob = log(rand);
    ratio = logLf_star - logLf;
    if prob > ratio
        reject = reject + 1;
    else
        f = fstar;
        logLf = logLf_star;
    end 
end

for j = 1:encoded_len
    temp = f(str(j));
    decoded(j) = string(temp);
end
end





