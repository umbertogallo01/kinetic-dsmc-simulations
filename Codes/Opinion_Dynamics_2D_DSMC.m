%% Ex 4_op - DIRECT SIMULATION MONTE CARLO (Kinetic model for opinion dynamics)
% Considerare lo stesso probelma dell'esercizio 4, ma in dimensione 2


clear
clc
close all


% Parametri del problema
N = 800;
dx = 0.09;
a = -4;
b = 4;
T = 300; 
mu=1;


% Parametri della dinamica delle opinioni (questa volta abbiamo che le
% opinioni son distribuite in un rettangolo [−n1_x*R,n2_x*R]x[−n1_y*R,n2_y*R]
n1x = 3;
n2x = 3;
n1y = 3;
n2y = 3;
R = 1;


% Definizione del set di random samples iniziali distribuiti uniformemente
% in Ω = [xmin,xmax]x[ymin,ymax] = [−n1_x*R,n2_x*R]x[−n1_y*R,n2_y*R]: 
xmin = -n1x*R;
xmax = n2x*R;
ymin = -n1y*R;
ymax = n2y*R;
V0 = [xmin,ymin] + [(xmax - xmin), (ymax - ymin)] .* rand([N,2]); % Per ogni riga abbiamo l'opinione (xi,yi)


% Direct Simulation Monte-Carlo Method
[V,t]=DSMC2(V0,N,R,mu,T);


% Riscostruzione della distribuzione delle opinioni al tempo T
[f_app,x,y] = reconstruct2([a,b],[a,b],dx,dx,V(:,:,end),N);
f_app = reshape(f_app,length(x),length(y)); % organize u_app as a matrix


% Visualizzazione grafica dei risultati

% Plot 2D delle opinioni al tempo t=0
figure(1)
subplot(1,2,1)
p1 = plot(V0(:,1),V0(:,2));

p1.Marker = 'o';
p1.MarkerFaceColor = "#0072BD";
p1.MarkerSize = 3;
p1.LineStyle = 'none';

xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
xlabel 'x'
ylabel 'y'
title 'Opinioni al tempo t=0 in 2D'


% Plot 3D della Distribuzione delle opinioni al tempo t=0
subplot(1,2,2)
f0=@(x,y) 1./(((n1x+n2x)*R)*((n1y+n2y)*R)) .*(x>=-n1x*R & x<=n1y*R & y>=-n1y*R & y<=n2y*R);

[X,Y] = meshgrid(x,y);
F0 = f0(X,Y);   % Valuto f0 sulla griglia

s1=surf(X,Y,F0);

s1.FaceAlpha = 0.5;
%s1.EdgeColor = 'none';
s1.FaceColor = 'interp';
xlabel('x'); ylabel('y'); zlabel('f0(x,y)')
clim([0,1]);
xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
zlim([0 1]);
xlabel('x'); ylabel('y'); zlabel('f(x,y,0)')
title 'Dsitribuzione delle opinioni al tempo t=0 in 3D'


% Plot 2D delle opinioni al tempo t=T
figure(2)
subplot(1,2,1)
p2 = plot(V(:,1,end),V(:,2,end));

p2.Marker = 'o';
p2.MarkerSize = 3;
p2.MarkerFaceColor = "#0072BD";
p2.LineStyle = 'none';

xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
xlabel 'x'
ylabel 'y'
title 'Opinioni al tempo t=T in 2D'


% Plot 3D della Distribuzione delle opinioni al tempo t=T
subplot(1,2,2)
s2 = surf(x,y,f_app);
s2.FaceAlpha = 0.5;
s2.EdgeColor = 'none';
s2.FaceColor = 'interp';
clim([0 max(f_app,[],'all')]);
xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
zlim([0 max(f_app,[]+1,'all')]);
zlabel 'f(x,y,t)'
title 'Distribuzione delle opinioni al tempo t=T in 3D'


% Animazione dell'evoluzione in tempo delle opinioni in 2D + 
% Animazione dell'evoluzione in tempo della distribuzione delle opinioni in 3D

figure(3)

% Grafico 2D
subplot(1,2,1)
p3 = plot(V(:,1,1),V(:,2,1));

p3.Marker = 'o';
p3.MarkerSize = 3;
p3.MarkerFaceColor = "#0072BD";
p3.LineStyle = 'none';

xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
xlabel 'x'
ylabel 'y'

title (['Evoluzione nel tempo delle opinioni (t= ' num2str(t(1)) ')'])

[ft_app,x,y] = reconstruct2([a,b],[a,b],dx,dx,V(:,:,1),N);
ft_app = reshape(ft_app,length(x),length(y));


% Grafico 3D
subplot(1,2,2)
s4 = surf(x,y,ft_app);
s4.FaceAlpha = 0.4;
s4.EdgeColor = 'none';
s4.FaceColor = 'interp';

clim([0 max(ft_app,[],'all')]);
xlim([-n1x*R-1 n2x*R+1])
ylim([-n1y*R-1 n2y*R+1])
zlim([0 max(ft_app,[],'all')+1]);
zlabel 'f(x,y,t)'

title (['Evoluzione nel tempo della distribuzione delle opinioni (t= ' num2str(t(1)) ')'])


waitforbuttonpress
for i=2:3:size(V,3)

    p3.XData = V(:,1,i);
    p3.YData = V(:,2,i);

    ft_app = reconstruct2([a,b],[a,b],dx,dx,V(:,:,i),N);
    ft_app = reshape(ft_app,length(x),length(y));
    s4.ZData = ft_app;

    drawnow;
    subplot(1,2,2)
    title (['Evoluzione nel tempo della distribuzione delle opinioni (t= ' num2str(t(i)) ')'])

    subplot(1,2,1)
    title (['Evoluzione nel tempo delle opinioni (t= ' num2str(t(i)) ')'])

    pause(0.01);

end



% Calcolo dei momenti: definizione delle osservabili phi=1, phi=x e
% phi=x^2/2
f0 = @(x) ones(size(x,1),size(x,2));
f1 = @(x) x;
f2 = @(x) 0.5*x.^2;


% Approssimazione dei moment tramite il metodo Monte Carlo
n=1;
m0=[];
m1=[];
m2=[];

while t(n)<T
    m0(:,n)=MC(V(:,:,n),N,f0);
    m1(:,n)=MC(V(:,:,n),N,f1);
    m2(:,n)=MC(V(:,:,n),N,f2);
    n=n+1;
end
m0(:,n)=MC(V(:,:,n),N,f0);
m1(:,n)=MC(V(:,:,n),N,f1);
m2(:,n)=MC(V(:,:,n),N,f2);


% Visualizzazione dell'evoluzione nel tempo dei momenti
figure(4)
p4 = plot3(t,m0(1,:),m0(2,:),t,m1(1,:),m1(2,:),t,m2(1,:),m2(2,:));
grid on

p4(1).LineWidth = 1.3;
p4(2).LineWidth = 1.3;
p4(3).LineWidth = 1.3;

p4(1).Color = "#E53333";
p4(2).Color = "#33B333";
p4(3).Color = "#FF9933";

xlabel 'Tempo'
ylabel 'Momento Prima Componente'
zlabel 'Momento Seconda Componente'
xlim([0 T]);
legend({'Momento zero','Momento primo','Momento secondo'})
title 'Evoluzione temporale dei momenti'




%% Funzioni 


function I = MC(X,N,g)

    G = g(X);               % Valuta la funzione g in tutti i punti della matrice X
    I = (1/N)*sum(G,1);     % Somma gli elementi per colonne

end




function [F,x,y] = reconstruct2(I1,I2,dx,dy,V,N)
% input
% - I1: intervallo delle componenti x (I1=[x1,x2])
% - I2: intevrallo delle componenti y (I2=[y1,y2])
% - dx,dy: passi di discretizzazione per I1 e I2 (risp) (dx deve dividere |I1| e dy deve dividere |I2|)
% - V: matrice Nx2 di random samples (Xi,Yi)
% - ker: variabile per la scelta della funzione kernel
%    1.  φ_∆x(x) = 1/∆x∆y, if |x| ≤ ∆x/2 e |y| ≤ ∆y/2 
%        φ_∆x(x) =  0, else
% - N : numero di samples (colonne di V)
%
% output 
% - X,Y: matrici con la discretizzazione di I1XI2 di passo dx (griglia)
% - F: matrice con i valori approssimati di f nella griglia di nodi (X(j),Y(i))

x = I1(1):dx:I1(2);     % discretizzazione dell' intervallo I1 con passo dx
y = I2(1):dy:I2(2);     % discretizzazione dell' intervallo I2 con passo dx
[X,Y] = meshgrid(x,y);  % matrici con le coordinate dei punti della griglia (X(i),Y(j))
F=[];

kernel = @(x,y) 1/(dx*dy).*(x>=(-dx/2)&x<=(dx/2) & y>=(-dy/2)&(y<=(dy/2))); 

A=[X(:),Y(:)];  % leggo le matrici X e Y per colonna in modo da costruirmi un vettore di N^2 righe e due colonne che sono i punti della griglia

for i=1:size(A,1)
    d=A(i,:)-V; % per ogni punto della griglia, calcolo la differenza tra il punto e i samples
    F=[F; (1/N)*sum(kernel(d(:,1),d(:,2)))];
end

% In questo modo in F dovrei avere tutti i valori approssimati
% della densità nei punti della griglia scritti colonna. Una volta usciti
% dobbiamo riorganizzarli in una matrice length(x) x length(y)

end




function [V,t] = DSMC2(V0,N,R,mu,T)
% input
% - V0: matrice con i random samples (X0,Y0)
% - N: numero di agenti
% - mu: parametro dell'equazione (=1)
% - R: parametro dell'equazione (bounded confidence)
% - T: istante di tempo finale
%
% output  
% - V: MATRICE 3D con evoluzione in tempo dei random samples (da
%      t=0 a t=T). La pagina n di V (matrice Nx2) contiene le opinioni al
%      tempo n 
% - t: vettore con istanti di tempo


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

    while t(n) < T

        Vnew = V(:,:,n);

  
        for i=1:N
            % con probabilità mu*dt seleziono un altro agente uniformemente
            % con cui interagirà l'agente i
            U=rand;
         
            if (U < mu*dt) 
                j=randi([1,N]);

                % L'agente j interagisce con l'agente i solo se
                % ||Vi-Vj||<=R

                if (norm(Vnew(i,:)-Vnew(j,:)) <= R) % calcolo la differenza tra la riga i e la riga j
                    W=rand();
                    if W<=0.5
                        Vnew(i,:) = p1*Vnew(i,:)+q1*Vnew(j,:);
                    else
                        Vnew(i,:)= p2*Vnew(i,:)+q2*Vnew(j,:);
                    end
               end

           end

        end

        V = cat(3,V,Vnew); % aggiungo V_new come nuova pagina a V
        n = n+1;
    end

end





