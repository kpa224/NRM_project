%% Belver project Group 2 (Part1 :Modelling)
%-------------------------------------------

%% import data, define variables
clc 
clear 

A = xlsread('02Belver.xlsx', 1); 

% import streamflow data
n = A(:, 5);
%figure; plot (n)

%devide streamflow dataset
n_cal = n(1:18*365);
n_val = n(18*365+1: 27*365);

% identfying some variables
N = length (n_cal); 
t = [ 1 : N ]'              ;  % indices t = 1 , 2 , ... , 365 , 366 , 367 , ...
T = 365;
tt = repmat( [1:365]' , N/T, 1 ) ;

%deseasonalize the calibration streamflow set
[ mi , m ] = moving_average( n_cal , T , 5 ); % moving average of streamflow
[ sigma2 , s2 ] = moving_average( ( n_cal - m ).^2 , T , 5 ) ;
sigma           = sigma2 .^ 0.5;                        
s               = s2 .^ 0.5;        
x = ( n_cal - m ) ./ s ; % x = deseasonalized calibration streamflow 

% deseasonalize the validation streamflow dataset
[ mi_val , m_val ] = moving_average( n_val , T , 5 ) ;
[ sigma2_val , s2_val ] = moving_average( ( n_val - m_val ).^2 , T , 5 ) ;
s_val = s2_val .^ 0.5 ;
xval = ( n_val - m_val ) ./ s_val ; %deseasonalzed validation streamflow

% import precipitation
p = A(:,4);
%check autocorrelation between flow and pecepitation
figure; correlogram(n,p,10);

%devide percepitation dataset
p_cal = p(1:18*365);
p_val = p(18*365+1: 27*365);

% deseasonalize precipitation for calibration dataset
[ mi_p , m_p ] = moving_average( p_cal , T , 5 ); % calculate moving average of calibration precipitation
[sigma2_p , s2_p ] = moving_average( (p_cal-m_p).^2 , T , 5 );
s_p = (s2_p).^0.5; 

u = (p_cal-m_p) ./ s_p;

% deseasonalize the validation set of precipitation
[ mi_pval , m_pval ] = moving_average( p_val , T , 5 );
[sigma2_pval , s2_pval ] = moving_average( (p_val-m_pval).^2 , T , 5 );
s_pval = (s2_pval).^0.5; 
u_val = (p_val-m_pval) ./ s_pval;



% import temperature
Tem = A(:, 6);
%check autocorrelation between flow and temperature
figure; correlogram(n,Tem,10);
%devide Temperature dataset
Tem_cal = Tem(1:18*365);
Tem_val = Tem(18*365+1: 27*365);


% deseasonalize Temperature for calibration dataset
[ mi_Tem , m_Tem ] = moving_average( Tem_cal , T , 5 ); % calculate moving average of calibration precipitation
[sigma2_Tem , s2_Tem ] = moving_average( (Tem_cal-m_Tem).^2 , T , 5 );
s_Tem = (s2_Tem).^0.5; 

c = (Tem_cal-m_Tem) ./ s_Tem;

% deseasonalize the validation set of Temperature
[ mi_Temval , m_Temval ] = moving_average( Tem_val , T , 5 );
[sigma2_Temval , s2_Temval ] = moving_average( (Tem_val-m_Temval).^2 , T , 5 );
s_Temval = (s2_Temval).^0.5; 
c_val = (Tem_val-m_Temval) ./ s_Temval;










%% statistics
% n_min   = min(n) ;
% n_max   = max(n) ;
% n_range = n_max - n_min ;
% n_mean  = mean(n) ;
% n_var   = var(n) ;

%% plot all the years
figure ; 
plot( t , n_cal ) ; 
xlabel( 'time t [days]' ) ; ylabel( 'inflow a_t [m^3/s]' )

%% reshape on single years --> cyclostationary statistics

figure;
plot( tt , n_cal , '.' ) ; 
xlabel( 'time t (1 year)' ) ; 
ylabel( 'inflow a_t [m^3/s]' );

Q = reshape(n_cal,T,N/T); % matrix of day by day streamflow (every column is a year)

Cm = mean(Q,2) ; % cyclostationary mean without moving average
% Cv = var(Q,2)  ;
hold on
plot (Cm,'r','LineWidth',2);

%% plotting moving average
hold on;
plot(mi, 'g'); legend ('inflow' ,'Cyclostationary mean' , 'moving average')

%% compute and plot periodic std

% figure ; plot( tt , n_cal , '.' ) ; hold on
% plot( [ 1 : T ] , mi , 'r', 'LineWidth',2  )
% legend('observed flow','moving cyclo mean');
% xlabel( 'time t (1 year)' );


%% plot the deseasonalized the inflow


figure ; plot( x ) ; xlabel( 'time t' ) ; ylabel( 'deseasonalized inflow x_t [m^3/s]' )
figure ; plot( tt , x ,'m.' ) ; xlabel( 'time t (1 year)' ) ; ylabel( 'deseasonalized inflow x_t [m^3/s]' )
% COMMENT : periodicity has been (partially) removed

%% autocorrelation 
% check the autocorrelation of raw data and deseasonalized data: if the
% they show a high autocorrelation, it means they can be modeled
% figure ; correlogram( n_cal , n_cal , 20 ) ; xlabel('k') ; ylabel('r_k')
% figure ; correlogram( x , x , 20 ) ; xlabel('k') ; ylabel('r_k')

% COMMENT : it is not white  (obvious) 


%% AR(1) calibration
%---------------------------------------------------

% identify and calculate parameter
y     = x( 2 : end )          ;
M     = [ x( 1  : end - 1 ) ] ;
theta = M \ y                 ;  

% forecast the inflow on calibration dataset with found parameter
x_    = M * theta             ;
x_    = [ x( 1 ) ; x_ ]       ;
n_    = x_ .* s + m           ;

% compare real inflow with forecast
figure ; 
plot( [ n_cal n_ ] , '.-' );
xlabel( 'time t [days]' ) ; ylabel( '[m^3/s]' ) ; legend( 'observed' , 'predicted' )

% calculate error  and evaluate autocorrelation
e_1 = n_cal( 2 : end ) - n_( 2 : end );
figure ; correlogram( e_1 , e_1 , 20 ) ; xlabel('k') ; ylabel('r_k')

% Mean Squared Error (MSE)
Qc = mean( e_1.^2 )
% coefficient of determination (Rt2)
R2_AR1_Cal = 1 - sum( e_1.^2 ) / sum( (n_cal( 2 : end )- m(2:end)).^2 )
         
%% AR(1) Validation
%---------------------------------------------------

% forecast with AR(1) on validation dataset
Mval    = xval( 1  : end - 1 ) ;
xval_   = Mval * theta         ;
xval_   = [ xval( 1 ) ; xval_ ]  ;
n_val_   = xval_ .* s_val + m_val   ;

% error betweeen real dataset and forecast (validation) and evaluate
% autocorrelation
eval = n_val( 2 : end ) - n_val_( 2 : end );
figure ; correlogram( eval , eval , 20 ) ; xlabel('k') ; ylabel('r_k')

% calculate mean squared error and coefficient of determination
Qval_1 = mean( eval.^2 )                                                                                
R2_AR1_Val = 1 - sum( (n_val( 2 : end ) - n_val_( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 )


%% AR(2) calibration 
%---------------------------------------------------

% identify and calculate parameters 
M2     = [ x( 1  : end - 2 ) , x( 2  : end - 1 ) ] ;
theta2  = M2 \ x( 3 : end )     ;

% forecast calibration dataset and reseasonalize
x_2    = M2 * theta2 ;
x_2    = [ x( 1:2 ) ; x_2 ] ;
n_2    = x_2 .* s + m     ;

% calculate error on calibration dataset and evaluate autocorrelation
e_2 = n_cal( 3 : end ) - n_2( 3 : end );
figure ; correlogram( e_2 , e_2 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qc_2 = mean( e_2.^2 )
R2_AR2_Cal = 1 - sum( (n_cal( 3 : end ) - n_2( 3 : end )).^2 ) / sum( (n_cal( 3 : end )- m(3:end)).^2 ) ;   

%% AR(2) validation
%---------------------------------------------------

% predict streamflow in validation period using the already deseasonalized
% validation dataset
M2  = [ xval( 1  : end - 2 ) , xval( 2  : end - 1 ) ] ;
xval_2 = M2 * theta2 ;
xval_2 = [ xval( 1:2 ) ; xval_2 ]         ;
n_val_2 = xval_2 .* s_val + m_val            ; % reseasonalize

% calculate error in validation period and evaluate autocorrelation
eval_2 = n_val( 3 : end ) - n_val_( 3 : end );
figure ; correlogram( eval_2 , eval_2 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qval_2 = mean( ( n_val( 3 : end ) - n_val_2( 3 : end ) ).^2 )
R2_AR2_Val = 1 - sum( (n_val( 3 : end ) - n_val_2( 3 : end )).^2 ) / sum( (n_val( 3 : end )-m_val(3:end)).^2 ) ;   



%% AR(4) Calibration
%---------------------------------------------------

% identify and calculate parameters 
M4     = [ x( 1  : end - 4 ) , x( 2  : end - 3 ) ,x( 3 : end - 2) ,x( 4: end - 1) ] ;
theta4  = M4 \ x( 5 : end )     ;

% forecast calibration dataset and reseasonalize
x_4    = M4 * theta4 ;
x_4    = [ x( 1:4 ) ; x_4 ] ;
n_4    = x_4 .* s + m     ;

% calculate error on calibration dataset and evaluate autocorrelation
e_4 = n_cal( 5 : end ) - n_4( 5 : end );
figure ; correlogram( e_4 , e_4 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qc_4 = mean( e_4.^2 )
R2_AR4_Cal = 1 - sum( (n_cal( 5 : end ) - n_4( 5 : end )).^2 ) / sum( (n_cal( 5 : end )- m(5:end)).^2 ) ;   

%% AR(4) Validation
%---------------------------------------------------

% validation dataset
M4  = [ xval( 1  : end - 4 ) , xval( 2  : end - 3 ), xval(3 : end -2) , xval(4 : end - 1) ] ;
xval_4 = M4 * theta4 ;
xval_4 = [ xval( 1:4 ) ; xval_4 ]         ;
n_val_4 = xval_4 .* s_val + m_val            ; % reseasonalize

% calculate error in validation period and evaluate autocorrelation
eval_4 = n_val( 5 : end ) - n_val_( 5 : end );
figure ; correlogram( eval_4 , eval_4 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qval_4 = mean( ( n_val( 5 : end ) - n_val_4( 5 : end ) ).^2 )
R2_AR4_Val = 1 - sum( (n_val( 5 : end ) - n_val_4( 5 : end )).^2 ) / sum( (n_val( 5 : end )-m_val(5:end)).^2 ) ;   


%% AR(6) Calibration
%---------------------------------------------------

% identify and calculate parameters 
M6     = [ x( 1  : end - 6 ) , x( 2  : end - 5 ) ,x( 3 : end - 4) ,x( 4: end - 3) ,x(5 : end-2),x(6 : end-1) ] ;
theta6  = M6 \ x( 7 : end )     ;

% forecast calibration dataset and reseasonalize
x_6    = M6 * theta6 ;
x_6    = [ x( 1:6 ) ; x_6 ] ;
n_6    = x_6 .* s + m     ;

% calculate error on calibration dataset and evaluate autocorrelation
e_6 = n_cal( 7 : end ) - n_6( 7 : end );
figure ; correlogram( e_6 , e_6 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qc_6 = mean( e_6.^2 )
R2_AR6_Cal = 1 - sum( (n_cal( 7 : end ) - n_6( 7 : end )).^2 ) / sum( (n_cal( 7 : end )- m(7:end)).^2 ) ;  

%% AR (6) Validation
%---------------------------------------------------

% validation dataset
M6  = [ xval( 1  : end - 6 ) , xval( 2  : end - 5 ), xval(3 : end -4) , xval(4 : end - 3),xval(5 :end-2) ,xval(6 :end-1) ] ;
xval_6 = M6 * theta6 ;
xval_6 = [ xval( 1:6 ) ; xval_6 ]         ;
n_val_6 = xval_6 .* s_val + m_val            ; % reseasonalize

% calculate error in validation period and evaluate autocorrelation
eval_6 = n_val( 7 : end ) - n_val_6( 7 : end );
figure ; correlogram( eval_6 , eval_6 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qval_6 = mean( ( n_val( 7 : end ) - n_val_6( 7 : end ) ).^2 )
R2_AR6_Val = 1 - sum( (n_val( 7 : end ) - n_val_6( 7 : end )).^2 ) / sum( (n_val( 7 : end )-m_val(7:end)).^2 ) ;   

%% AR (7) Calibration
%---------------------------------------------------

% identify and calculate parameters 
M7     = [ x( 1  : end - 7 ) , x( 2  : end - 6 ) ,x( 3 : end - 5) ,x( 4: end - 4) ,x(5 : end-3),x(6 : end-2), x(7: end-1) ] ;
theta7  = M7 \ x( 8 : end )     ;

% forecast calibration dataset and reseasonalize
x_7    = M7 * theta7 ;
x_7    = [ x( 1:7 ) ; x_7 ] ;
n_7    = x_7 .* s + m     ;

% calculate error on calibration dataset and evaluate autocorrelation
e_7 = n_cal( 8 : end ) - n_7( 8 : end );
figure ; correlogram( e_7 , e_7 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qc_7 = mean( e_7.^2 )
R2_AR7_Cal = 1 - sum( (n_cal( 8 : end ) - n_7( 8 : end )).^2 ) / sum( (n_cal( 8 : end )- m(8:end)).^2 ) ;  

%% AR (7) Validation
%---------------------------------------------------

% validation dataset
M7  = [ xval( 1  : end - 7 ) , xval( 2  : end - 6 ), xval(3 : end -5) , xval(4 : end - 4),xval(5 :end-3) ,xval(6 :end-2) ,xval(7:end-1) ] ;
xval_7 = M7 * theta7 ;
xval_7 = [ xval( 1:7 ) ; xval_7 ]         ;
n_val_7 = xval_7 .* s_val + m_val            ; % reseasonalize

% calculate error in validation period and evaluate autocorrelation
eval_7 = n_val( 8 : end ) - n_val_7( 8 : end );
figure ; correlogram( eval_7 , eval_7 , 20 ) ; xlabel('k') ; ylabel('r_k')

% mean square error and coef of determination
Qval_7 = mean( ( n_val( 8 : end ) - n_val_7( 8 : end ) ).^2 )
R2_AR7_Val = 1 - sum( (n_val( 8 : end ) - n_val_7( 8 : end )).^2 ) / sum( (n_val( 8 : end )-m_val(8:end)).^2 ) ;   


%WE TERMINATE HERE AND CHOOSE AR(6) BECAUSE THE MEAN SQUARE ERROR (J) OF VALIDATION SET
%SHOWED AN INCREASE IN AR(7) WHICH MEANS (OVER FITTING) OR (OVER PARAMETERIZATION)

%% Comparisons Between AR(p) models in validation.

% comparison of simulated flow
%figure; plot([n_val n_val_ n_val_2 n_val_4 n_val_6 n_val_7]);legend('obs','AR1', 'AR2','AR4' ,'AR6' ,'AR7');
n_AR6= [n_6 ; n_val_6];

figure; plot([n n_AR6 ]);legend('obs','AR6'); xlabel('t (days)');ylabel('stream flow (m3/s)')

% comparison of model accuracy
figure; bar([R2_AR1_Val R2_AR2_Val R2_AR4_Val R2_AR6_Val R2_AR7_Val]); 
%comparison of model error
figure; bar([Qval_1 Qval_2 Qval_4 Qval_6 Qval_7])
%J = [ Qval_1 Qval_2 Qval_4 Qval_6 Qval_7 ]
%AR= [1 2 4 6 7]
%figure ; plot (AR ,J); xlabel ('p') ,ylabel ('MSE')







%%
% plot precipitation and cyclostationary mean

figure;
plot( tt , p_cal , '.' ) ; 
xlabel( 'time t (1 year)' ) ; 
ylabel( 'precipitation a_t [mm]' );

Q_p = reshape(p_cal,T,N/T); % matrix of precipitation day by day (every column is a year)

Cm_p = mean(Q_p,2) ;

hold on
plot (Cm_p,'r','LineWidth',2);

%and on the same curve you plot the moving average for precipitation
hold on;
plot(mi_p, 'g', 'LineWidth', 3);

%plot deasonalized precipitation on tt scale
figure;
plot (tt, u, '.'); xlabel('time (1 year)'); ylabel('deseasonalized precipitation [mm]');






%% ARX(1,1)(with Percepitation) proper - calibration
%---------------------------------------------------

% identify and compute parameters
y = x(2:end) ;
M = [ x(1:end-1) u(1:end-1) ] ;  
theta_pro = M \ y;

% predict deseasonalized streamflow in calibration period
x_arx_pro = [ x(1); M * theta_pro ] ;
% seasonalize predicted streamflow
n_arx_pro = x_arx_pro .* s + m ;

% calculate error on calibration dataset and evaluate autocorrelation
e_arx_pro = n_cal( 2  : end ) - n_arx_pro( 2 : end );
figure ; correlogram( e_arx_pro , e_arx_pro , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qc_Arx_pro = mean( e_arx_pro.^2 )
R2_ARX_pro_Cal    = 1 - sum( (n_cal( 2 : end ) - n_arx_pro( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )

%% ARX(1,1)(with Percepitation) proper - validation
%---------------------------------------------------

% predict streamflow in validation period 
Mval = [ xval(1:end-1) u_val(1:end-1) ] ;  
x_val_pro = [xval(1); Mval*theta_pro];
n_val_pro = x_val_pro .* s_val + m_val; % reseasonalized predicted streamflow

% calculate error on validation dataset and evaluate autocorrelation
e_val_pro = n_val( 2  : end ) - n_val_pro( 2 : end );
figure ; correlogram( e_val_pro , e_val_pro , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qval_Arx_pro = mean( e_val_pro.^2 )
R2_ARX_pro_Val    = 1 - sum( (n_val( 2 : end ) - n_val_pro( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 )


n_arx_p_pro_full= [n_arx_pro ;n_val_pro] %simulate full stream flow timeseries for plotting purpose

%% ARX(1,1)(with Percepitation) improper - calibration
%---------------------------------------------------

%identify and compute parameters
M = [ x(1:end-1) u(2:end) ] ; 
y = x(2:end) ;
theta_imp    = M \ y 
% predict deseasonalized streamflow in calibration period
x_arx_imp = [ x(1); M * theta_imp ] ;
% seasonalize predicted streamflow
n_arx_imp = x_arx_imp .* s + m ;

% calculate error on calibration dataset and evaluate autocorrelation
e_arx_imp = n_cal( 2  : end ) - n_arx_imp( 2 : end );
figure ; correlogram( e_arx_imp , e_arx_imp , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qc_Arx_impro = mean( e_arx_imp.^2 )
R2_ARX_impro_Cal    = 1 - sum( (n_cal( 2 : end ) - n_arx_imp( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )


%% ARX(1,1)(with Percepitation) improper - validation
%---------------------------------------------------
% predict streamflow in validation period 
Mval = [ xval(1:end-1) u_val(2:end) ] ;  
x_val_imp = [xval(1); Mval*theta_imp];
n_val_imp = x_val_imp .* s_val + m_val; % reseasonalized predicted streamflow

% calculate error on validation dataset and evaluate autocorrelation
e_val_imp = n_val( 2  : end ) - n_val_imp( 2 : end );
figure ; correlogram( e_val_imp , e_val_imp , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qval_Arx_impro = mean( e_val_imp.^2 )
R2_ARX_impro_Val    = 1 - sum( (n_val( 2 : end ) - n_val_imp( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 )

%% ARX (1,1) (With Temperature) (proper)- Calibration
%---------------------------------------------------

% identify and compute parameters
y = x(2:end) ;
M = [ x(1:end-1) c(1:end-1) ] ;  
theta_pro_temp = M \ y;

% predict deseasonalized streamflow in calibration period
x_arx_pro = [ x(1); M * theta_pro_temp ] ;
% seasonalize predicted streamflow
n_arx_pro = x_arx_pro .* s + m ;

% calculate error on calibration dataset and evaluate autocorrelation
e_arx_pro_Temp = n_cal( 2  : end ) - n_arx_pro( 2 : end );
figure ; correlogram( e_arx_pro_Temp , e_arx_pro_Temp , 20 ) ; xlabel('k') ; ylabel('r_k')



%mean square error and the coef of determination
Qc_Arx_pro_Temp = mean( e_arx_pro_Temp.^2 )
R2_ARX_pro_Temp_Cal    = 1 - sum( (n_cal( 2 : end ) - n_arx_pro( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )

%% ARX(1,1)(with Temperature) proper - validation
%---------------------------------------------------

% predict streamflow in validation period 
Mval = [ xval(1:end-1) c_val(1:end-1) ] ;  
x_val_pro = [xval(1); Mval*theta_pro_temp];
n_val_pro = x_val_pro .* s_val + m_val; % reseasonalized predicted streamflow

% calculate error on validation dataset and evaluate autocorrelation
e_val_pro_Temp = n_val( 2  : end ) - n_val_pro( 2 : end );
figure ; correlogram( e_val_pro_Temp , e_val_pro_Temp , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qval_Arx_pro_Temp = mean( e_val_pro_Temp.^2 )
R2_ARX_pro_Temp_Val    = 1 - sum( (n_val( 2 : end ) - n_val_pro( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 )

%% ARX(1,1)(with Temperature) improper - Calibration
%---------------------------------------------------

%identify and compute parameters
M = [ x(1:end-1) c(2:end) ] ; 
y = x(2:end) ;
theta_imp_Temp    = M \ y 
% predict deseasonalized streamflow in calibration period
x_arx_imp = [ x(1); M * theta_imp_Temp ] ;
% seasonalize predicted streamflow
n_arx_imp = x_arx_imp .* s + m ;

% calculate error on calibration dataset and evaluate autocorrelation
e_arx_imp_Temp = n_cal( 2  : end ) - n_arx_imp( 2 : end );
figure ; correlogram( e_arx_imp_Temp , e_arx_imp_Temp , 20 ) ; xlabel('k') ; ylabel('r_k')

%mean square error and the coef of determination
Qval_Arx_impro_Temp = mean( e_arx_imp_Temp.^2 )
R2_ARX_impro_Temp_Cal    = 1 - sum( (n_cal( 2 : end ) - n_arx_imp( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )

%% ARX (1,1) (With Temperature) improper - Validation
%---------------------------------------------------

% predict streamflow in validation period 
Mval = [ xval(1:end-1) c_val(2:end) ] ;  
x_val_imp = [xval(1); Mval*theta_imp];
n_val_imp = x_val_imp .* s_val + m_val; % reseasonalized predicted streamflow

% calculate error on validation dataset and evaluate autocorrelation
e_val_imp_Temp = n_val( 2  : end ) - n_val_imp( 2 : end );
figure ; correlogram( e_val_imp_Temp , e_val_imp_Temp , 20 ) ; xlabel('k') ; ylabel('r_k')


%mean square error and the coef of determination
Qval_Arx_impro_Temp = mean( e_val_imp_Temp.^2 )
R2_ARX_impro_Temp_Val    = 1 - sum( (n_val( 2 : end ) - n_val_imp( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 )



%% ANN Artificial Neural Network - PROPER
%---------------------------------------------------

% define input and target vectors
X = [ x(1:end-1) u(1:end-1) c(1:end-1) ]' ;
Y = x(2:end)' ;

% % define and train neural network
% net = newff(X,Y,3) ; % initialization of ANN
% net = train( net, X, Y ) ;
% 
% % simulate output (streamflow)
% Y_ = sim( net, X ) ;
% Y_pro = [ x(1); Y_' ] ;
% n_ann_pro = Y_pro .* s + m ; % reseasonalize predicted streamflow
% 
% % coef of determination
% R2_ANN_pro = 1 - sum( (n_cal( 2 : end ) - n_ann_pro( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )


N_runs = 20 ;
R2_i = zeros(N_runs,1);
for i = 1:N_runs
    net_i = newff(X,Y,4) ; % initialization of ANN
    net_i = train( net_i, X, Y ) ;
    Y_ = sim( net_i, X ) ;
    Y_ = [ x(1); Y_' ] ;
    n_ann_i = Y_ .* s + m ;

    R2_i(i) = 1 - sum( (n_cal( 2 : end ) - n_ann_i( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 );
    if R2_i(i) >= max(R2_i)
        net_opt_pro = net_i ;
    end
end

% % simulate output (streamflow)
Y_ = sim( net_opt_pro, X ) ;
Y_pro = [ x(1); Y_' ] ;
n_ann_pro = Y_pro .* s + m ; % reseasonalize predicted streamflow

R2_ANN_pro = max(R2_i);
clear R2_i;
clear net_i;

%% validation on proper ANN 
%---------------------------------------------------

% define input and output on validation dataset
X_val = [xval(1:end-1) u_val(1:end-1) c_val(1:end-1)]';
Y_val = xval(2:end)';

% predict output and reaseasonalize
Y_val_= sim(net_opt_pro, X_val);
Y_val_ = [xval(1) Y_val_]';
n_ann_val = Y_val_ .* s_val + m_val;

%mean square error and the coef of determination

e_val_ANN_pro = n_val( 2  : end ) - n_ann_val( 2 : end );
figure ; correlogram( e_val_ANN_pro , e_val_ANN_pro , 20 ) ; xlabel('k') ; ylabel('r_k')


Qval_ANN_Pro = mean( e_val_ANN_pro.^2 )

R2_ANN_pro_val = 1 - sum( (n_val( 2 : end ) - n_ann_val( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 );

n_ann_pro_full= [n_ann_pro ;n_ann_val] ; %simulate full stream flow timeseries for plotting purpose



%% ANN Artificial Neural Network - IMPROPER
%---------------------------------------------------

% define input and target vectors
X = [ x(1:end-1) u(2:end) c(2:end) ]' ;
Y = x(2:end)' ;

% % define and train neural network
% net_imp = newff(X,Y,3) ; % initialization of ANN
% net_imp = train( net_imp, X, Y ) ;
% 
% % simulate output (streamflow)
% Y_ = sim( net_imp, X ) ;
% Y_imp = [ x(1); Y_' ] ;
% n_ann_imp = Y_imp .* s_q + m_q ; % reseasonalize predicted streamflow
% 
% % coef of determination
% R2_ANN_imp = 1 - sum( (n_cal( 2 : end ) - n_ann_imp( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 )

N_runs = 20 ;
R2_i = zeros(N_runs,1);
for i = 1:N_runs
    net_i = newff(X,Y,4) ; % initialization of ANN
    net_i = train( net_i, X, Y ) ;
    Y_ = sim( net_i, X ) ;
    Y_ = [ x(1); Y_' ] ;
    n_ann_i = Y_ .* s + m ;

    R2_i(i) = 1 - sum( (n_cal( 2 : end ) - n_ann_i( 2 : end )).^2 ) / sum( (n_cal( 2 : end )-m(2:end) ).^2 );
    if R2_i(i) >= max(R2_i)
        net_opt_impro = net_i ;
    end
end

R2_ANN_impro = max(R2_i);
clear R2_i;
clear net_i;

%% validation on improper ANN 

% define input and output on validation dataset
X_val = [xval(1:end-1) u_val(2:end) c_val(2:end)]';
Y_val = xval(2:end)';

% predict output and reaseasonalize
Y_val_= sim(net_opt_impro, X_val);
Y_val_ = [xval(1) Y_val_]';
n_ann_val = Y_val_ .* s_val + m_val;

%mean square error and the coef of determination

e_val_ANN_imp = n_val( 2  : end ) - n_ann_val( 2 : end );


Qval_ANN_imPro = mean( e_val_ANN_imp.^2 )

R2_ANN_impro_val = 1 - sum( (n_val( 2 : end ) - n_ann_val( 2 : end )).^2 ) / sum( (n_val( 2 : end )-m_val(2:end) ).^2 );


%% Comparisons between the forecasted stream flow in ARX proper with 
%percipitaiton ,ANN proper and AR(6)



figure ;plot ([n n_AR6 n_arx_p_pro_full n_ann_pro_full]) ;legend ('obs','AR6','ARX(1,1) Proper','ANN Proper');
hold on;title ('predicted stream flow in diffirent models') ;xlabel('time (days)'); ylabel('stream flow m3/s')







