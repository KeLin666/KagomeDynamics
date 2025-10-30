%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  2D Schrodinger Equation (Continuum Model)  %%%%%%%
%%%%%%  Boundary Condition: X-periodic,Y-periodic  %%%%%%%
%%%%%% Input: Potential, Grid, number of eigenvalue %%%%%%
%%%%%%     Output: Eigenvalue and Eigenfunction    %%%%%%%
%%%%%% Note: numeig should be smaller than size(V) %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,BZS]=SE2Deig_KXKY(V,Nkx,Nky,dx,dy,kx,ky,guess_value,numeig)
% Nkx equal to length of kx, Nky equal to length of ky
hbar=1.05457182e-34;
me=9.10938356e-31;
coeff=hbar^2/2/me;

Ne=5*Nkx*Nky;
E=zeros(length(kx),length(ky),numeig);
BZS=zeros(length(kx),length(ky),numeig,Nkx*Nky);

for indexkx=1:length(kx)
    
    [num2str(indexkx/length(kx)*100),'%']

    KX=kx(indexkx);
    
    for indexky=1:length(ky)
        
        KY=ky(indexky);
        I=zeros(Ne,1);
        J=zeros(Ne,1);
        H=zeros(Ne,1);
        C1=-(-1/dx/dx-1/dy/dy-KX*KX/2-KY*KY/2)*coeff;
        C2=-(1/2/dx/dx+1i*KX/2/dx)*coeff;
        C3=-(1/2/dx/dx-1i*KX/2/dx)*coeff;
        C4=-(1/2/dy/dy+1i*KY/2/dy)*coeff;
        C5=-(1/2/dy/dy-1i*KY/2/dy)*coeff;
        
        p=0;

        %%% Hamiltonian: 4 Corner Points

        % H[i,j]=[1,1]
        i=1;
        j=1;
        k=j+(i-1)*Nkx;
        I(p+1:p+5)=[k,k,k,k,k];
        J(p+1:p+5)=[k,k+1,k+Nkx-1,k+Nkx,k+Nkx*Nky-Nkx];
        H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
        p=p+5;
        
        % H[i,j]=[1,Nx]
        i=1;
        j=Nkx;
        k=j+(i-1)*Nkx;
        I(p+1:p+5)=[k,k,k,k,k];
        J(p+1:p+5)=[k,k-Nkx+1,k-1,k+Nkx,k+Nkx*Nky-Nkx];
        H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
        p=p+5;
        
        % H[i,j]=[Ny,1]
        i=Nky;
        j=1;
        k=j+(i-1)*Nkx;
        I(p+1:p+5)=[k,k,k,k,k];
        J(p+1:p+5)=[k,k+1,k+Nkx-1,k-Nkx*Nky+Nkx,k-Nkx];
        H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
        p=p+5;
        
        % H[i,j]=[Ny,Nx]
        i=Nky;
        j=Nkx;
        k=j+(i-1)*Nkx;
        I(p+1:p+5)=[k,k,k,k,k];
        J(p+1:p+5)=[k,k-Nkx+1,k-1,k-Nkx*Nky+Nkx,k-Nkx];
        H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
        p=p+5;
        
        %%% Hamiltonian: Boundary Points
        i=1;
        for j=2:Nkx-1
            k=j+(i-1)*Nkx;
            I(p+1:p+5)=[k,k,k,k,k];
            J(p+1:p+5)=[k,k+1,k-1,k+Nkx,k+Nkx*Nky-Nkx];
            H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
            p=p+5;
        end
        
        i=Nky;
        for j=2:Nkx-1
            k=j+(i-1)*Nkx;
            I(p+1:p+5)=[k,k,k,k,k];
            J(p+1:p+5)=[k,k+1,k-1,k-Nkx*Nky+Nkx,k-Nkx];
            H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
            p=p+5;
        end
        
        j=1;
        for i=2:Nky-1
            k=j+(i-1)*Nkx;
            I(p+1:p+5)=[k,k,k,k,k];
            J(p+1:p+5)=[k,k+1,k+Nkx-1,k+Nkx,k-Nkx];
            H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
            p=p+5;
        end
        
        j=Nkx;
        for i=2:Nky-1
            k=j+(i-1)*Nkx;
            I(p+1:p+5)=[k,k,k,k,k];
            J(p+1:p+5)=[k,k-Nkx+1,k-1,k+Nkx,k-Nkx];
            H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
            p=p+5;
        end
        
        %%% Hamiltonian: Ordinary Points
        for i=2:Nky-1
            for j=2:Nkx-1
                k=j+(i-1)*Nkx;
                I(p+1:p+5)=[k,k,k,k,k];
                J(p+1:p+5)=[k,k+1,k-1,k+Nkx,k-Nkx];
                H(p+1:p+5)=[C1+V(i,j),C2,C3,C4,C5];
                p=p+5;
            end
        end

        % Sparse Matrix note - I,J: Index of position in Sparse Matrix; H: Value of [i,j] in I and J
        Hamiltonian=sparse(I,J,H,Nkx*Nky,Nkx*Nky); 
        [bzs,bzz]=eigs(Hamiltonian,numeig,guess_value);
        eigvalue=diag(bzz);
        [d0,id0]=sort(real(eigvalue),'ascend');
        
        for indexeig=1:numeig
            E(indexkx,indexky,indexeig)=d0(indexeig);
        end
        
        for indexeig=1:numeig
            BZS(indexkx,indexky,indexeig,:)=bzs(:,id0(indexeig));
        end
        
    end
end

end

