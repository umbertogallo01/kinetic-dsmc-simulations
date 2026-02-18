%% Ex 2 - DIRECT SIMULATION MONTE CARLO 
% 1. Compute u(x,t) solution of the simple ODE
%               ∂tu(x,t) = C*u(x,t)
% with initial condition u_0(x) and where C∈R is a constant,
% using the Direct Simulation Monte Carlo with a doubling/removing technique. 
% Assume that u_0(x) is a probability density, in particular consider both the 
% a. uniform density
% b. standard normal density


clear
clc 
close all

% Parametri del problema
N = 50000;
dx = 0.1;
a = -4;     % Assumiamo di considerare la soluzione per x in [-4,4]
b = 4;
C=-1;
T=5;        % Tempo finale


% Definizione della densità di probabilità iniziale u0(x)
initial_density=menu('Scegliere la densità iniziale u0(x) tra:','Uniforme ~ U(0,1)','Normale standard ~ N(0,1)','Pearson ~ P(2,1,2,10)');

switch initial_density
    case 1
        u0=@(x) 1.*(x<=1 & x>=0);   % distribuzione uniforme in [0,1]
        X0=rand([N,1]);             % vettore colonna

    case 2
        mu = 0;
        sigma = 1;
        u0=@(x) normpdf(x,mu,sigma);  % distribuzione normale standard
        X0 = normrnd(mu,sigma,[N,1]);

    case 3
        mu = 2;
        sigma = 1;
        skew = 2;
        kurt = 10;
        X0 = pearsrnd(mu,sigma,skew,kurt,N,1);
     
        %g = @(x,mu,sigma)(1./(sigma.*sqrt(2*pi))).*exp((-(x-mu).^2)./(2.*sigma.^2));
        u0 = @(x) pearspdf(x,mu,sigma,skew,kurt);
end


% Direct Simulation Monte Carlo
X=DSMC(X0,C,N,T);


% Ricostruzione della soluzione
kernel=2;
[u_appr,x] = reconstruction(a,b,N,dx,X,kernel);

if initial_density==3 % pearson distr
    [u_appr,x] = reconstruction(0,10,N,dx,X,kernel);
end


% Warning: la soluzione ottenuta dalla funzione di ricostruzione non è una
% densità di probabilità, quindi va moltiplicata per la massa
u_appr=u_appr.*exp(C*T);


% Soluzione esatta

if initial_density==3
    y=0:1.e-4:10;
    a=0;
    b=10;
else
    y =-4:1.e-6:4;
end

u_esatta=u0(y).*exp(C*T);
u0=u0(y);
% figure(1)
% plot(y,u0)

% Visualizzazione grafica dei risultati
figure(2)
p=plot(y,u_esatta,x,u_appr);
p(1).Color='#08519c';
p(1).LineWidth = 1.5;
p(2).LineStyle = '--';
p(2).Marker='o';
p(2).MarkerFaceColor = "#e41a1c";
p(2).Color = "#e41a1c";
p(2).LineWidth = 2;

legend('Soluzione esatta','Soluzione approssimata con DSMC');
axis([a b 0 max([max(u_esatta),max(u_appr)])]);


%% Funzioni usate

% Direct-Simulation Monte Carlo Method
function X=DSMC(X0,C,N,T)
    
    % input
    % - X0: random samples iniziali
    % - C: parametro dell'equazione (tasso di rilassamento)
    % - N: numero di samples
    % - T: istante di tempo finale
    %
    % output 
    % - X: vettore con i samples generati con la tecnica doubling/removing

    
    % Definizione del time-step dt che deve verificare |C*dt| ≤ 1 (combinazione convessa)
    dt=0.5 * (1/abs(C));

    t = 0:dt:T;
    if t(end) < T       % non ricopro [0,Tfin] con dt definito sopra
        Nt = length(t)+1;
    
    else                % ricopro [0,Tfin] con dt
        Nt = length(t);
    end
    dt = T/(Nt-1);

    
    t=0;
    while t<T
        % Seleziono uniformemente un indice tra gli N
        k=randi([1 N]);     

        % Genero un random sample U1 ~ U(0,1)
        U1=rand();

        % Distinguo i vari casi
        if C>0

            if U1<C*dt
                % Doubling
                X0=[X0;X0(k)];
                N=N+1;
            end

        else

            if U1<-C*dt
                % Removing
                X0(k)=[];
                N=N-1;
            end

        end

        t=t+dt;
    end

    X=X0;
end



function [f,x] = reconstruction(a,b,N,dx,X,kernel)

% input
% - a,b: estremi intervallo [a,b]
% - dx: passo di discretizzazione per [a,b] (dx deve dividere b-a)
% - X: vettore di random samples  
% - ker: variabile per la scelta della funzione kernel
%    1.  φ_∆x(x) = 1/∆x, if |x| ≤ ∆x/2 
%        φ_∆x(x) =  0, else
%    2.  φ_∆x(x) = 1/∆x*B_2(x/∆x) (definizione di B2 nel testo)
%
% output 
% - x: vettore con la discretizzazione dell'intervallo di [a,b]
% - f: vettore con i valori approssimati di f nei nodi di discretizzazione x(j) di [a,b] 

x = a:dx:b;                 % discretizzazione dell' intervallo [a,b] con passo dx
f = zeros(1,length(x));     % vettore per ricostruzione di f

    switch kernel

        case 1 
            
            kernel = @(y) (1/dx)*((y<=(dx/2)) & y>=-(dx/2)); 
            
            A = x-X;
            F = kernel(A);
            f = (1/N)*(sum(F));
    
        case 2
 
            a = @(y) 0.75 - abs(y).^2;
            b = @(y) 0.5 * (abs(y)-(3/2)).^2;
            kernel = @(y) 1/dx * a(y/dx) .* (abs(y/dx)<=(0.5)) + 1/dx * b(y/dx) .* ((abs(y/dx)<=(1.5)) & abs(y/dx)>(0.5));

            A = x-X;   
            F = kernel(A);
            f = (1/N)*(sum(F));
    end

end
