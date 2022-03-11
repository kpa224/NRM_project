%% Belver project Group 2 (Part2 :Planning and management)
%---------------------------------------------------------


%% Capacity design via Sequent Peak Analysis

clc 
clear 
close all

A = xlsread('02Belver.xlsx', 1); 

% import streamflow data
n = A(:, 5);
clear A;
% monthly mean flow
qMonth = dailyToMonthly(n, 27); % m3/s

% target release = downstream water demand w
w = prctile(n,50) ;% m3/s  %taken as the median (50th percentile) of the inflow
%w_t = ones (size(n))*w
%figure; plot (n)
%hold on ; plot (w_t,'--','LineWidth',2) ;legend ('streamflow','water demand') ; xlabel('time t (days)');ylabel('stream flow (m3/s)')

% Sequent Peak Analysis
deltaT = 3600*24*[31 28 31 30 31 30 31 31 30 31 30 31]';
Q = qMonth(:).*repmat(deltaT,27,1) ; % m3/month
W = w*ones(size(Q)).*repmat(deltaT,27,1) ; % m3/month
K = zeros(size(Q));
K(1) = 0;

for t = 1:length(Q)
    K(t+1) = K(t) + W(t) - Q(t) ;
    if K(t+1) < 0
        K(t+1) = 0 ;
    end        
end

figure; plot( K )
Kopt = max(K)  % m3

DH = 112  %m *dam height %comparing the storage with other real reservoirs (Moehne ,Germany) almost same storage with dam height = 40m


clear deltaT 
clear K
clear Q
clear qMonth
clear t
clear W

%% simulation of water reservoir under regulation

%natural river
param.nat.S = Kopt/DH ;  % assuming cylindrical shape of the reservoir
param.nat.beta = max (n) /DH ; % assuming max release as max natural inflow ,beta is the natural flow line slope
param.nat.alpha = 1 ;  %assuming Linear Max/Min release
param.nat.h0 = 0;   %h0=hmin 

%% Alternative 1
%-------------


% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0 ;
param.reg.h_max = DH ;
param.reg.h1 = 7 ;
param.reg.h2 = 60 ;
param.reg.m1 = 70 ;
param.reg.m2 = 50 ;

% test regulated level-discharge relationship
h_test = [ 0 : 1 : 120 ] ;
r_test = regulated_release( param, h_test ) ;
r_max= param.nat.beta*(h_test)
figure; plot(h_test,r_test)
hold on; plot (h_test, r_max, 'r--') ;title('ALTERNATIVE 1') ;xlabel('h (m)') ;ylabel ('r (m3/s)') ;legend ('Policy','Max release')

% simulation of lake dynamics

n_ = [ nan; n ] ; % for time convention
h_init = 56 ; % initial condition %assumed half dam height


[s, h, r] = simulate_lake( n_, h_init, param ) ;

%figure; plot( h )
%figure; plot (s)
%figure; plot (r)

% Perfomance

h1 = h(2:end); 
r1 = r(2:end);
w = param.reg.w ;
h_flo = 0.8 * DH ;  %flooding risk threshold is 80% of the dam height.

%WATER SUPPLY INDICATORS

% daily average squared deficit
def = max( w-r1, 0 ) ;
Jirr_1 = mean( def.^2 )

% reliability
Irel_1 = sum( r1 >= w ) / length(r1)

% vulnerability
Ivul_1 = sum(def) / sum( r1 < w )

%FLOODING INDICATORS

%(annual average number of days with flooding events)
Ny = length(h1)/365
Jflo_1 = sum( h1>h_flo )/Ny

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 

% IE1 = number of low pulses
IE_LP_1 = sum( r1 < LP ) / Ny
% IE2 = number of high pulses
IE_HP_1 = sum( r1 > HP ) / Ny




%% Alternative (2)
%-----------------

% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0 ;
param.reg.h_max = DH ;
param.reg.h1 = 5 ;
param.reg.h2 = 95  ;
param.reg.m1 = 20 ;
param.reg.m2 = 45 ;

% test regulated level-discharge relationship
h_test = [ 0 : 1 : 120 ] ;
r_test = regulated_release( param, h_test ) ;
figure; plot(h_test,r_test)
hold on; plot (h_test, r_max, 'r--') ;title('ALTERNATIVE 2') ;xlabel('h (m)') ;ylabel ('r (m3/s)') ;legend ('Policy','Max release')


% simulation of lake dynamics

n_ = [ nan; n ] ; % for time convention
h_init = 56 ; % initial condition %assumed half dam height


[s, h, r] = simulate_lake( n_, h_init, param ) ;

%figure; plot( h )
%figure; plot (s)
%figure; plot (r)

% Perfomance

h2 = h(2:end); 
r2 = r(2:end);
w = param.reg.w ;
h_flo = 0.8 * DH ;  %flooding risk threshold is 80% of the dam height.

%WATER SUPPLY INDICATORS

% daily average squared deficit
def = max( w-r2, 0 ) ;
Jirr_2 = mean( def.^2 )

% reliability
Irel_2 = sum( r2 >= w ) / length(r2)

% vulnerability
Ivul_2 = sum(def) / sum( r2 < w )

%FLOODING INDICATORS

%(annual average number of days with flooding events)
Ny = length(h2)/365
Jflo_2 = sum( h2>h_flo )/Ny

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 

% IE1 = number of low pulses
IE_LP_2 = sum( r2 < LP ) / Ny
% IE2 = number of high pulses
IE_HP_2 = sum( r2 > HP ) / Ny

%% Alternative (3)
%-----------------

% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0 ;
param.reg.h_max = DH ;
param.reg.h1 = 4 ;
param.reg.h2 = 86 ;
param.reg.m1 = 200 ;
param.reg.m2 = 200 ;

% test regulated level-discharge relationship
h_test = [ 0 : 1 : 120 ] ;
r_test = regulated_release( param, h_test ) ;
figure; plot(h_test,r_test)
hold on; plot (h_test, r_max, 'r--') ;title('ALTERNATIVE 3') ;xlabel('h (m)') ;ylabel ('r (m3/s)') ;legend ('Policy','Max release')


% simulation of lake dynamics

n_ = [ nan; n ] ; % for time convention
h_init = 56 ; % initial condition %assumed half dam height

[s, h, r] = simulate_lake( n_, h_init, param ) ;

%figure; plot( h )
%figure; plot (s)
%figure; plot (r)

% Perfomance

h3 = h(2:end); 
r3 = r(2:end);
w = param.reg.w ;
h_flo = 0.8 * DH ;  %flooding risk threshold is 80% of the dam height.

%WATER SUPPLY INDICATORS

% daily average squared deficit
def = max( w-r3, 0 ) ;
Jirr_3 = mean( def.^2 )

% reliability
Irel_3 = sum( r3 >= w ) / length(r3)

% vulnerability
Ivul_3 = sum(def) / sum( r3 < w )

%FLOODING INDICATORS

%(annual average number of days with flooding events)
Ny = length(h3)/365
Jflo_3 = sum( h3>h_flo )/Ny

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 

% IE1 = number of low pulses
IE_LP_3 = sum( r3 < LP ) / Ny
% IE2 = number of high pulses
IE_HP_3 = sum( r3 > HP ) / Ny

%% Alternative 4
%----------------

% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0 ;
param.reg.h_max = DH ;
param.reg.h1 = 5 ;
param.reg.h2 = 87 ;
param.reg.m1 = 100 ;
param.reg.m2 = 150 ;

% test regulated level-discharge relationship
h_test = [ 0 : 1 : 120 ] ;
r_test = regulated_release( param, h_test ) ;
figure; plot(h_test,r_test)
hold on; plot (h_test, r_max, 'r--') ;title('ALTERNATIVE 4') ;xlabel('h (m)') ;ylabel ('r (m3/s)') ;legend ('Policy','Max release')


% simulation of lake dynamics

n_ = [ nan; n ] ; % for time convention
h_init = 56 ; % initial condition %assumed half dam height


[s, h, r] = simulate_lake( n_, h_init, param ) ;

%figure; plot( h )
%figure; plot (s)
%figure; plot (r)

% Perfomance

h4 = h(2:end); 
r4 = r(2:end);
w = param.reg.w ;
h_flo = 0.8 * DH ;  %flooding risk threshold is 80% of the dam height.

%WATER SUPPLY INDICATORS

% daily average squared deficit
def = max( w-r4, 0 ) ;
Jirr_4 = mean( def.^2 )

% reliability
Irel_4 = sum( r4 >= w ) / length(r4)

% vulnerability
Ivul_4 = sum(def) / sum( r4 < w )

%FLOODING INDICATORS

%(annual average number of days with flooding events)
Ny = length(h4)/365
Jflo_4 = sum( h4>h_flo )/Ny

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 

% IE1 = number of low pulses
IE_LP_4 = sum( r4 < LP ) / Ny
% IE2 = number of high pulses
IE_HP_4 = sum( r4 > HP ) / Ny

%% Alternative (5)
%-----------------

% regulated level-discharge relationship:
param.reg.w = w ;
param.reg.h_min = 0 ;
param.reg.h_max = DH ;
param.reg.h1 = 5 ;
param.reg.h2 = 95 ;
param.reg.m1 = 20 ;
param.reg.m2 = 45 ;

% test regulated level-discharge relationship
h_test = [ 0 : 1 : 120 ] ;
r_test = regulated_release( param, h_test ) ;
figure; plot(h_test,r_test)
hold on; plot (h_test, r_max, 'r--') ;title('ALTERNATIVE 5') ;xlabel('h (m)') ;ylabel ('r (m3/s)') ;legend ('Policy','Max release')


% simulation of lake dynamics

n_ = [ nan; n ] ; % for time convention
h_init = 56 ; % initial condition %assumed half dam height


[s, h, r] = simulate_lake( n_, h_init, param ) ;

%figure; plot( h )
%figure; plot (s)
%figure; plot (r)

% Perfomance

h5 = h(2:end); 
r5 = r(2:end);
w = param.reg.w ;
h_flo = 0.8 * DH ;  %flooding risk threshold is 80% of the dam height.

%WATER SUPPLY INDICATORS

% daily average squared deficit
def = max( w-r5, 0 ) ;
Jirr_5 = mean( def.^2 )

% reliability
Irel_5 = sum( r5 >= w ) / length(r5)

% vulnerability
Ivul_5 = sum(def) / sum( r5 < w )

%FLOODING INDICATORS

%(annual average number of days with flooding events)
Ny = length(h5)/365
Jflo_5 = sum( h5>h_flo )/Ny

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 

% IE1 = number of low pulses
IE_LP_5 = sum( r5 < LP ) / Ny
% IE2 = number of high pulses
IE_HP_5 = sum( r5 > HP ) / Ny

%comparisons

%Jirr= [Jirr_1 Jirr_2 Jirr_3 Jirr_4 Jirr_5]
%Jflo= [Jflo_1 Jflo_2 Jflo_3 Jflo_4 Jflo_5]
%figure ; plot (Jirr , Jflo ,'d') ; xlabel ('Jirr') ; ylabel ('Jflod') ;legend ({'Policy1' ,'Policy2','Policy3','Policy4','Policy5'}) ;

%% ploting policies
figure ;plot (Jirr_1,Jflo_1,'d') ;
hold on ; plot(Jirr_2,Jflo_2,'d') ; 
hold on ; plot (Jirr_3,Jflo_3,'d') ; 
%hold on ; plot (Jirr_4 ,Jflo_4 ,'d')
%hold on ; plot (Jirr_5 , Jflo_5 ,'d')
hold on; legend ('Policy1' , 'Policy2' ,'Policy3') ;xlabel('Jirr') ;ylabel ('Jflod'); title('Objectives')

%% natural River (Alternative 0 No Dam)


%WATER SUPPLY INDICATORS

% reliability
Irel_nat = sum( n >= w ) / length(n)

% vulnerability
def = max( w-n, 0 )
Ivul_nat = sum(def) / sum( n < w )

% daily average squared deficit
Jirr_nat = mean( def.^2 )

%ENVIRONMENTAL INDICATORS
LP = prctile( n, 25 ) % thresholds are defined over the inflow
HP = prctile( n, 75 ) % thresholds are defined over the inflow 
Ny= length (h)/365
% IE1 = number of low pulses
IE_LP_nat = sum( n < LP ) / Ny
% IE2 = number of high pulses
IE_HP_nat = sum( n > HP ) / Ny



% Jflod_nat (cannot be calculated because we don't have a time series of water
% level in the river + we cannot use the same threshold. 

