function [E,BZS]=band_structure(Tx,Ty,Nkx,Nky,V_unit,numeig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%         Band structure           %%%%%%%%%%
%%%%%%%%%%        Writen by Ke Lin          %%%%%%%%%%
%%%%%%%%%% Heller Group, Harvard University %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Set-up part  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Unit and constant
hbar=1.05457182e-34;
me=9.10938356e-31;
coeff=hbar^2/2/me;
eV=1.60218e-19;

% real space grid set up
Nx_unit=length(V_unit);    % make sure Nx equal to the size of unit cell
x_vector=linspace(-Tx/2+Tx/Nx_unit/2,Tx/2-Tx/Nx_unit/2,Nx_unit);
dx=x_vector(2)-x_vector(1);

Ny_unit=length(V_unit);    % make sure Ny equal to the size of unit cell
y_vector=linspace(-Ty/2+Ty/Ny_unit/2,Ty/2-Ty/Ny_unit/2,Ny_unit);
dy=y_vector(2)-y_vector(1);

% momentum space grid set up 

% Nkx=15; % size of k vector, recommend value: 31
kx=linspace(-pi/Tx,pi/Tx,Nkx);
% Nky=15; % size of k vector, recommend value: 31
ky=linspace(-pi/Ty,pi/Ty,Nky);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  Main calculation part  %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

guess_value=eV;

[E,BZS]=SE2Deig_KXKY(V_unit,Nx_unit,Ny_unit,dx,dy,kx,ky,guess_value,numeig);


end