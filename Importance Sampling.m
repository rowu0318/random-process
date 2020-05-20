function [est_prob,est_se] = Importance_sampling()

trials = 100000;  % number of trials
n = 20; % 20 steps of length 1

bias = [1,3,4,6,8,9,10,11,12,16,60,70,80,99]; % biasing parameters
na = length(bias);

nbins = 100;    % binning parameters
maxx = 20;
dx = maxx/nbins;
xbins = (0+dx/2:dx:maxx-dx/2);

binavg = zeros(1,nbins);    % vectors to save results
cdfavg = zeros(1,nbins);
cdcavg = zeros(1,nbins);
misvar = zeros(1,nbins);
nums = zeros(1,nbins);

sz = get(0,'ScreenSize');
wid = sz(3);
hyt = sz(4);

% loop through biasing parameters
for ka = 1:na
    dist = ones(1,trials);
    ratios = ones(na,trials);   
    % loop through 20 steps
    for j = 2:n
        x = rand(1,trials);
        cos_phi = 2.*x.^(1/bias(ka)) - ones(1,trials);
        
        for kk = 1:na           
            pstar = bias(kk).*((cos_phi+1).^(bias(kk)-1))/(2^(bias(kk)));
            p = 1/2;
            ratio = p./pstar;
            ratios(kk,:) = ratios(kk,:).*ratio;
        end
        
        fprintf(1,' bias %4i, pass %4i\n',ka,j)
        % update the distance using cosine theorem
        dist = 1 + dist.^2 + 2.*dist.*cos_phi;
        dist = sqrt(dist);
    end
    weight=1./(sum(1./ratios,1));
    % bin the results
    bintmp = zeros(1,nbins);
    for j = 1:nbins
        % indicator function for every bin
        indx = find((j-1)*dx<dist & dist<=j*dx);
        indc = find(dist<=(j*dx));
        indr = find(dist>(j*dx));
        if ~isempty(indx)
            binavg(j) = binavg(j)+sum(weight(indx))/trials;
            bintmp(j) = sum(weight(indx))/trials;
        end
        nums(j) = nums(j)+length(weight(indx));
        
        estim = sum(weight(indx))/trials;  
        % Compute variance; get contribution from each
        % biasing distribution
        tmpweight = zeros(size(weight));
        tmpweight(indx) = weight(indx);
        misvar(j) = misvar(j)+sum((tmpweight-estim*ones(size(tmpweight))).^2)/(trials*(trials-1));
        
        % Compute cdf and 1-cdf
        cdfavg(j)=cdfavg(j)+sum(weight(indc))/trials;
        cdcavg(j)=cdcavg(j)+sum(weight(indr))/trials;
    end
    f1=figure(1);
    set(f1,'Position',[0.025*wid 0.525*hyt wid/3 hyt/3]);
    semilogy(xbins,bintmp/dx,'linewidth',2.0,'MarkerSize',12);
    a=gca;
    set(a,'linewidth',1.0,'FontSize',14);
    xlabel('z','FontSize',16);
    ylabel('weighted probabilities');
    drawnow;
    hold on;
end

f2=figure(2);
set(f2,'Position',[0.35*wid 0.525*hyt wid/3 hyt/3]);
% Asymptotic result
xx = 0.1:0.01:20;
rou = 1./tanh(xx)- 1./xx;
asy_p = rou.*xx./(maxx*sqrt((pi/(2*n^3))*(1 - rou.^2 - 2.*rou./xx))).*exp(-n.*rou.*xx);
asy_p = asy_p.*((sinh(xx)./xx).^n);
semilogy(rou*n,asy_p,'r*',xbins,binavg/dx,'b-','linewidth',2.0);
a=gca;
set(a,'linewidth',1.0,'FontSize',14);
xlabel('z','FontSize',16);
ylabel('weighted probability');
legend('Asymptotic','Estimated')

f3=figure(3);
set(f3,'Position',[0.35*wid 0.025*hyt wid/3 hyt/3]);
plot(xbins,sqrt(misvar)./binavg,'linewidth',2);
a=gca;
set(a,'linewidth',1.0,'FontSize',14);
xlabel('z','FontSize',16);
ylabel('coefficient of variation','FontSize',16);

est_prob = binavg/dx;
est_se = sqrt(misvar)./binavg;
end