%% Ex 1 - RECONSTRUCTION PROBLEM
% Write a function for the reconstruction of a probability density starting 
% from discrete samples Xk as:
%       f_N(xj) = 1/N*sum_all_j(φ_∆x(xj−Xk)), j=...-2,-1,0,1,2,...
% where xj= j∆x and the kernel φ can be chosen between:
%
% a. φ_∆x(x) = 1/∆x, if |x| ≤ ∆x/2 
%    φ_∆x(x) =  0, else
%
% b. φ_∆x(x) = 1/∆x*B_2(x/∆x),
%    where B2(x) = 3/4 −|x|^2, if |x| ≤ 0.5 
%          B2(x) = ((|x|−3/2)^2)/2, if 0.5 < |x| ≤ 1.5
%          B2(x) = 0, else     
% 
% Test the function by reconstructing a standard normal density on
% x∈[−4,4] choosing:
% - N = 500 samples 
% - ∆x = 0.4

clear
clc 
close all


% Parametri del problema
N = 100000;
dx = 0.4;
a = -4;
b = 4;


% Samples X(i) ~ N(0,1) for i=1,...,N
mu = 0;
sigma = 1;
X = normrnd(mu,sigma,[N,1]);


% Scelta nell'approssimazione della delta di Dirach
kernel=menu('Scegliere l''approssimazione della delta di Dirach:','1. φ_∆x(x) = 1/∆x*χ_[-∆x/2, ∆x/2])(x)','2. φ_∆x(x) = 1/∆x*B_2(x/∆x)','3. Poly6');


% Chiamo la funzione di ricostruzione per approssimare la distribuzione
% normale
[f_a,x]=reconstruction(a,b,N,dx,X,kernel);


% Definizione della distribuzione normale esatta
f=@(x) normpdf(x,mu,sigma);
xf = -4:1.e-6:4;
f=f(xf);


% Calcolo dell'errore di approssimazione (bisogna avere la stessa
% discretizzazione spaziale)
%err=norm(f-f_a,Inf)


% Visualizzazione grafica dei risultati
p=plot(xf,f,x,f_a);
p(1).Color='k';
p(2).LineStyle = '--';
p(2).Marker='o';
p(2).Color = "#008000";
p(2).MarkerFaceColor="#008000";
legend('Distriuzione normale esatta','Distribuzione normale approssimata');
title(['Ricostruzione della densità della normale standard con ker = ' num2str(kernel)]);
axis([-4 4 0 max([max(f),max(f_a)])]);



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
            
            A = x-X;            % Le colonne di A sono xj-X
            F = kernel(A);      % Valuta la matrice precedente elemento per elemento
            f = (1/N)*(sum(F));
    
        case 2
            a = @(y) 0.75 - abs(y).^2;
            b = @(y) 0.5 * (abs(y)-(3/2)).^2;
            kernel = @(y) 1/dx * a(y/dx) .* (abs(y/dx)<=(0.5)) + 1/dx * b(y/dx) .* ((abs(y/dx)<=(1.5)) & abs(y/dx)>(0.5));

            A = x-X;   
            F = kernel(A);
            f = (1/N)*(sum(F));


        % Consideriamo una terza approssimazione della delta di Dirac usando un polinomio di grado 6  
        case 3 
            kernel=@(y) (dx.^2-abs(y).^2).^3 .*(abs(y)<=dx/2);
            C=1/integral(kernel,-Inf,+Inf); % Costante di normalizzazine della funzione kernel
            kernel=@(y) C*(dx.^2-abs(y).^2).^3 .*(abs(y)<=dx/2);
           
            % figure(2)
            % y=a:0.05:b;
            % p2=plot(y,kernel(y));

            A = x-X;   
            F = kernel(A);
            f = (1/N)*(sum(F));
    end

end