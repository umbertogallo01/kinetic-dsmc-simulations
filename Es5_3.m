%% Ex 3 - DIRECT SIMULATION MONTE CARLO 2 (Goldstein-Taylor model)
% 1. Solve the Goldstein-Taylor model
%       ∂tf+(x,t) + c*∂xf+(x,t) = σ*(f−(x,t)−f+(x,t))
%       ∂tf−(x,t) − c*∂xf−(x,t) = σ*(f+(x,t)−f−(x,t))
% with the Direct Simulation Monte Carlo method.
% Let: ρ(x,t) = f+(x,t) + f−(x,t).
%
% Take:
% - σ = 1 
% - c = 1
% - initial condition:
%    ρ0(x) = 1, for −1 ≤ x ≤ 0, 
%    ρ0(x) = 0 elsewhere, 
%    with f+_0 (x) = f−_0 (x). 
% 
% Solve the model with:
% - N = 15000 samples
% - final time T = 0.38
% 
% Show ρ(x,T) and the distribution of the velocities.
%
% Consider also the case in which f+_0 (x) != f−_0 (x) and show that
% the asymptotic state f+_0 (x) = f−_0 (x) = 1/2 is reached either for 
% - longer times 
% - larger σ


clear
clc
close all


% Parametri del problema
N = 15000;     % numero di samples
dx = 0.05;
T = 0.38; 
c = 1; 
sigma = 1;  
kernel = 2;    % parametro per la funzione di ricostruzione


% Definizione delle condizioni iniziali:
% - le particelle sono distribuite inizialmente uniformemente in [a,b]
%
% - inizialmente abbiamo che f0+(x)=f0-(x), cioè la probabilità che le
%   particelle abiamo velocità +c o -c è la stessa. Successivamente proviamo
%   a vedere cosa succede quando sono diverse

a=-1;
b=0;
[X0,V0]=initial_condition(a,b,N,c);

% Definiamo le densità al tempo t=0
rho_0=@(y) 1.*(y>=-1 & y<=0);
V0_1= @(y) length(V0(V0==c)).*(y==c);
V0_2= @(y) length(V0(V0==-c)).*(y==-c);
y= -4:1.e-6:3;
rho_0=rho_0(y);
V0_1=V0_1(y)/N;
V0_2=V0_2(y)/N;


% Direct Simulation Monte Carlo Method
[X,V]=DSMC(X0,V0,sigma,c,N,T);


% Ricostruzione della soluzione

% Riscostruzione della densità delle particelle rho(x,t)
[rho_appr,x] = reconstruction(-4,3,N,dx,X,kernel);

% Ricostruzione delle distribuzioni delle velocità
V1 = @(v) length(V(V==c))/N .* (v==c);
V2 = @(v) length(V(V==-c))/N .* (v==-c);
v=x;
[~,index1]=min(abs(v-c));
[~,index2]=min(abs(v+c));
v(index1)=1;
v(index2)=-1;
V1 = V1(v);
V2 = V2(v);

% [v1_app,x1] = reconstruction(-4,3,length(V1),dx,V1,kernel); % distribuzione della velocità +C
% [v2_app,x2] = reconstruction(-4,3,length(V2),dx,V2,kernel); % distribuzione della velocità -C



% Visualizzazione grafica dei risultati

% Densità delle particelle al tempo t=0
figure(1);
p1 = plot(y,rho_0);
p1.LineWidth = 1;
p1.LineStyle = '-';
p1.Color = '#008000';
xlabel 'Posizioni'
title 'Densità iniziale delle particelle ρ(x,t=0)'


% Distribuzione delle velocità al tempo t=0
figure(2)
p2 = plot(y,V0_1,y,V0_2);
p2(1).LineWidth = 1;
p2(1).LineStyle = '-';
p2(1).Color = "#1E90FF";
p2(2).LineWidth = 1;
p2(2).LineStyle = '-';
p2(2).Color = "#FF7F50";
xlabel 'Velocità'
legend({'Particelle con V=c','Particelle con V=-c'})
title 'Distribuzione delle velocità al tempo t=0'


% Densità delle particelle al tempo finale t=T
figure(3)
p3=plot(x,rho_appr);
p3.LineWidth = 1;
p3.LineStyle = '-';
p3.Color ="#008000"';
xlabel 'Posizione';
title 'Approssimazione della densità delle particelle ρ(x,T)';


% Distribuzione delle velocità al tempo t=T
figure(4)
p4 = plot(v,V1,v,V2); 
p4(1).LineWidth = 1;
p4(1).LineStyle = '-';
p4(1).Color = "#1E90FF"; %rosso scuro
p4(2).LineWidth = 1;
p4(2).LineStyle = '-';
p4(2).Color = "#FF7F50"; %azzurro
xlabel 'Velocità'
legend({'Particelle con V=c','Particelle con V=-c'})
title 'Distribuzione delle velocità al tempo t=T'



%% Funzioni

function [X,V]=DSMC(X0,V0,sigma,c,N,T)
    % input
    % - (X0,V0): random samples iniziali
    % - c: velocità di propagazione delle particelle
    % - sigma: parametro dell'equazione (tasso di rilassamento)
    % - N: numero di samples
    % - T: istante di tempo finale
    %
    % output 
    % - X: vettore con le posizioni finali delle particelle al tempo finale
    % T
    % - V: vettore con le velocità delle particelle al tempo finale T 

    
    % Definizione del time-step dt che deve verificare |c*dt| ≤ 1 (combinazione convessa)
    dt=0.5 * (1/abs(c));

    t = 0:dt:T;
    if t(end) < T           % non ricopro [0,Tfin] con dt definito sopra
        Nt = length(t)+1;
    
    else                    % ricopro [0,Tfin] con dt
        Nt = length(t);
    end
    dt = T/(Nt-1);

    t=0;

    X=X0;
    V=V0;
    
    while t<T
        % Aggiornamento delle posizioni
        X=X+dt*V0;

        % Aggiornamento delle velocità per ogni particella
        for i=1:N

            % Definiamo un parametro estratto uniformemente per capire se
            % le particelle hanno interagito con il background oppure no
            q1=rand;

            if q1 < 1-exp(-2*sigma*dt)

                % Definiamo un altro parametro estratto uniformemente per capire se
                % lo stato delle particelle dopo l'interazione cambia
                % oppure no
                q2=rand;

                % con probabilità 1/2 pongo la velocità della particella i-esima pari a c
                if q2 < 0.5
                    V(i) = c;

                % con probabilità 1/2 pongo la velocità della particella i-esima pari a -c      
                else 
                    V(i) = -c;

                end   

            end

        end

        t=t+dt;
    end

end



function [X0,V0]=initial_condition(a,b,N,c)
    % input
    % - a,b: estremi dell'intervallo in cui si trovano le particelle
    % - N: numero di random samples
    % - c: parametro dell'equazione (velocità di propagazione)
    % - p: probabilità di una particella di assumere velocità c
    %       deve avere un valore tra 0 e 1
    %       es: p = 0 -> tutte le particelle con velocità -c
    %           p = 1 -> tutte le particelle con velocità c
    %           p = 0.5 -> velocità c e -c con stessa probabilità
    %
    % output
    % - X0: vettore con le posizioni delle particelle distribuite uniformemente in [a,b]
    % - V0: vettore con le velocità delle particelle (pari a -c o c)

    X0= a + (b-a) * rand([N,1]);

    % Per determinare le velocità, introduciamo una variabile p che
    % rappresenta la probabilità di avere velocità +c. A seconda del valore
    % di p possiamo avere varie situazioni (p=0.5 stessa probabilità +/- c)
    
    p=0.5;

    % Estraggo uniformemente N samples e confronto i valori con p. Se il
    % valore dell'estrazione è < p, allora assegno velocità +c, altrimenti
    % -c
    Samples=rand([N,1]);
    V0 = c*(2 * (Samples < p) - 1);

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
