%% Ex 5 - DIRECT SIMULATION MONTE CARLO (Kinetic model for traffic flow)
% Consider the traffic flow model with interactions with the background
%                   ∂tf(v,t) = ρint_(0,Vmax)(δ(v−v^∗)f(t,v∗)dv∗)−ρf(t,v)
% where the post interactional state 
% is:
%    v^∗ = v∗ + P(ρ)(VA−v∗), if v∗ < u
%    v^∗ = v∗ − (1−P(ρ))(v∗−VB ), v∗ > u.
%
% with: 
% - P(ρ) = 1−ρ,
% - VA = min{v∗+ ∆v,Vmax},
% - VB = max{u−∆v,0}, 
% - ∆v= 0.2
% - u(t) = int_(0,Vmax)(vf(v,t)dv) (the mean speed).
% 
% Take Vmax = 1.
%
% For a fine discretization of the density ρ ∈[0,1]
% 1. find the corresponding state f∞ (f for large times) of the kinetic model 
% with the Direct Simulation Monte Carlo method
%
% 2. show the relation ρ → u = int_(0,1)(vf∞(v)dv)


clc
clear all
close all


% Parametri del problema
N=1.e3;
a=0;
b=1;
T=50;  % Per avere una buona funzione u, T deve essere grande
Vmax = 1;
rho = 0.3; 
dv = 0.2;
dx=0.01;
kernel = 2; 


% Estrazione degli N random samples Vi_0 distribuiti come f0(v). Supponiamo
% che le velocità siano distribuite uniformemente tra [0,Vmax]
V0= Vmax .* rand([N,1]);

% Test 2: proviamo ad estrarre le velocità secondo una normale standard
% V0 = randn([N,1]);


% Direct Simulation Monte Carlo Method per l'approssimazione di f(v,t)
[V,t]=DSMC(V0,dv,rho,N,T,Vmax);


% Ricostruisco la funzione di densità al tempo t=T
[f_app,v]=reconstruction(a,b,N,dx,V,kernel);



% Visualizzazione grafica dei risultati

% Plot dell'approssimazione della soluzione dell'equazione cinetica f(v,T) 
% al tempo t=T
figure(1)
p1=plot(v,f_app);
title 'Approssimazione della distribuzione f(v,t) al tempo t=T'
xlabel 'Velocità'
ylabel 'f(v,t)'


% Trovare, per una fine discretizzazione della densità rho in [0,1], il
% corrispondente stato f∞ con il DSMC e mostrare la relazione  
% ρ → u = int_(0,1)(vf∞(v)dv)

rho=0:0.001:1;
g=@(v) v .* (v>=0 & v<=Vmax);
u_rho=[];   % inizializzo il vettore delle velocità media

for i=1:length(rho)

    % Calcolo i samples usando rho(i)
    [V,~]=DSMC(V0,dv,rho(i),N,T,Vmax);

    % Calcolo il valore della velocità media u(rho(i)) usando la
    % distribuzione precedente (in particolare i samples, dato che usiamo
    % il metodo Monte Carlo e sappiamo che i samples sono distribuiti come 
    % f∞(rho))
    I=MC(g,V,N);

    % Salvo il risultato in un vettore
    u_rho=[u_rho,I];

end


figure(2)
plot(rho,u_rho)
title 'Velocità media in funzione della densità dei veicoli'
xlabel 'Densità'
ylabel 'Velocità media u(rho)'




%% Funzioni

function [V,t]=DSMC(V0,dv,rho,N,T,Vmax)

    % Definizione del time-step dt che deve verificare mu*dt ≤ 1 (combinazione convessa)
    dt=0.1 * (1/rho);
    
    t = 0:dt:T;
    if t(end) < T           % non ricopro [0,Tfin] con dt definito sopra
        Nt = length(t)+1;
    
    else                    % ricopro [0,Tfin] con dt
        Nt = length(t);
    end
    dt = T/(Nt-1);
    t=0:dt:T;
    
    V=V0;
    n=1;

    % Definisco la funzione g(v) per il calcolo della velocità media 
    % tramite la formula di quadratura fornita da Monte Carlo
    g=@(v) v .* (v>=0 & v<=Vmax);


    % Definisco la funzione P(rho)=1-rho che rappresenta la probabilità di
    % accelerare
    P=@(rho) 1-rho;


    while t(n)<T
        
        % calcolo la velocità media u(t)
        u=MC(g,V,N);

        for i=1:N
            
            % Con probabilità pari a rho*dt cambio velocità
            U=rand;

            if U<rho*dt

                if V(i)<u
                    V_A=min(V(i)+dv,Vmax);
                    V(i)=V(i)+P(rho)*(V_A-V(i));

                else
                    V_B=max(u-dv,0);
                    V(i)=V(i)-(1-P(rho))*(V(i)-V_B);
                    
                end

            end

        end

        n=n+1;
    
    end

end



function I=MC(g,V,N)
    G=g(V);
    I=(1/N)*sum(G);
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

