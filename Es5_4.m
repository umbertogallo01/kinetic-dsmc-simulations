%% Ex 4.1 - DIRECT SIMULATION MONTE CARLO 3 (Kinetic model for opinion dynamics)
% Consider the opinion dynamics kinetic model
%           ∂tf(t,x) = int_Ω(int_Ω(η(y,z;R)K(x|y,z)f(t,y)f(t,z)dydz))
%                      -f(t,x)*int_Ω(η(x,z;R)f(t,z)dz)
% where:
% - K(x|y,z) = δ(x−(y+z)/2)
% - η(y,z;R) = χ_{|y−z|≤R}(y,z)
% describing compromise between two interacting agents.
% 
% 1. Solve the model up to a sufficiently large time 
% using the Direct Simulation Monte Carlo algorithm. 
% 
% Consider the initial probability distribution f_0 = f(0,x) as
% - the uniform distribution 
% - on the set Ω = [−n_1*R,n_2*R]
% - choose N = 10^4 samples from f_0(x)
%
% Show the evolution in time of the kinetic distribution f(t,x) taking:
% - R = 1
% - n1 = n2 = 1,2,3,4


clear
clc
close all


% Parametri del problema
N = 1.e4;
dx = 0.1;
a = -4;
b = 4;
T = 100; 
mu=1;
kernel = 1; % Parametro per la funzione di ricostruzione


% Parametri della dinamica delle opinioni
n1 = 2;
n2 = 2;
R = 1;


% Definizione del set di random samples iniziali distribuiti uniformemente
% in Ω = [−n_1*R,n_2*R]
xmin=-n1*R;
xmax=n2*R;
X0=xmin + (xmax-xmin) * rand([N,1]);


% Distribuzione delle opinioni al tempo t=0
f0=@(y) 1/((n1+n2)*R).*(y>=xmin & y<=xmax);
y= a:1.e-6:b;
f0=f0(y);


% Direct Simulation Monte-Carlo Method
[X,t]=DSMC1(X0,N,R,mu,T);


% Riscostruzione della distribuzione delle opinioni
[f_appr,x] = reconstruction(a,b,N,dx,X(:,end),kernel);


% Visualizzazione grafica dei risultati

% Grafico della distribuzione delle opinioni al tempo 0 + grafico della
% distribuzione delle opinioni al tempo t=T ottenuta tramite DSMC1
figure(1);
subplot(1,2,1)
p1 = plot(y,f0);
grid on
p1.LineWidth = 1;
p1.LineStyle = '-';
p1.Color = '#008000';
xlabel 'Opinioni'
ylabel 'f(x,0)'
title 'Dsitribuzione delle opinioni al tempo t=0'

subplot(1,2,2)
p2=plot(x,f_appr);
grid on
p2.LineWidth = 1;
p2.LineStyle = '-';
p2.Color = '#008000';
xlabel 'Opinioni'
ylabel 'f(x,T)'
title 'Dsitribuzione delle opinioni al tempo t=T'


% Evoluzione nel tempo della distribuzione delle opinioni
figure(2)
[ft_appr,x]=reconstruction(a,b,N,dx,X(:,1),kernel);
ft_appr=ft_appr';

for i=2:size(X,2)
    ft_appr_new = reconstruction(a,b,N,dx,X(:,i),kernel);
    ft_appr_new=ft_appr_new';
    ft_appr = [ft_appr,ft_appr_new];
end

c = contourf(t,x,ft_appr,'LineStyle','none');
colormap 'sky' %only on R2023a version
b = colorbar;
title 'Evoluzione nel tempo della distribuzione delle opinioni'
xlabel 'Tempo'
ylabel 'Opinioni'


% 2. Evoluzione in tempo dei momenti


% Calcolo dei momenti: definizione delle osservabili phi=1, phi=x e
% phi=x^2/2
f0 = @(x) ones(size(x,1),size(x,2));    % Costruisce una matrice di 1 con un numero di righe e colonne corrispondente alle dimensioni di x
f1 = @(x) x;
f2 = @(x) 0.5*x.^2;


% Approssimazione dei momenti tramite il metodo Monte Carlo
m0 = MC(X,N,f0);
m1 = MC(X,N,f1);
m2 = MC(X,N,f2);


% Visualizzazione dell'evoluzione nel tempo dei momenti
figure(3)
p3 = plot(t,m0,t,m1,t,m2);

grid on
xlabel 'Tempo'
ylabel 'Momenti'

p3(1).LineWidth = 1.3;
p3(2).LineWidth = 1.3;
p3(3).LineWidth = 1.3;

p3(1).Color = "#E53333";
p3(2).Color = "#33B333";
p3(3).Color = "#FF9933";

xlim([0 T]);
ylim([-0.4 1.2])
legend({'Momento zero','Momento primo','Momento secondo'})
title 'Evoluzione temporale dei momenti'


%% Funzioni

function I = MC(X,N,g)
% input
% - X: vettore di random samples (X(i) ~ f(x))
% - g: funzione (function handle) di cui voglio calcolare E[g(X)]
%
% output  
% - I: approssimazione dell'integrale di (gf)(x) (o di E[g(X)])

% X rappresenta una matrice che ha per colonne i samples al tempo t
    
    G = g(X);               % Valuta la funzione g in tutti i punti della matrice X
    I = (1/N)*sum(G,1);     % Somma gli elementi per colonne

end



function [V,t]=DSMC1(V0,N,R,mu,T)
    
    % Definizione dei parametri p1,q1,p2,q2 per gli stati post
    % interazionali
    p1=0.5;
    q1=0.5;
    p2=0.5;
    q2=0.5;

    % Definizione del time-step dt che deve verificare mu*dt ≤ 1 (combinazione convessa)
    dt=0.1 * (1/mu);
    
    t = 0:dt:T;
    if t(end) < T % non ricopro [0,Tfin] con dt definito sopra
        Nt = length(t)+1;
    
    else % ricopro [0,Tfin] con dt
        Nt = length(t);
    end
    dt = T/(Nt-1);
    t=0:dt:T;
    
    V=V0;
    n=1;

    while t(n)<T

        Vnew=V(:,n);

        for i=1:N
            % con probabilità mu*dt seleziono un altro agente uniformemente
            % con cui interagirà l'agente i
            U=rand;
    
            if U<mu*dt
                j=randi([1,N]);
                
                % L'agente j interagisce con l'agente i solo se
                % ||Vi-Vj||<=R
                if abs(Vnew(i)-Vnew(j))<=R
                    W=rand;
        
                    if W<0.5
                        Vnew(i)=p1*Vnew(i)+q1*Vnew(j);
                    else
                        Vnew(i)=p2*Vnew(i)+q2*Vnew(j);
                    end

                end

            end
    
        end

        V=[V Vnew];
        n=n+1;
    end

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
