function z=fp_wft2f(type,f,sigmax,wxl,wxi,wxh,sigmay,wyl,wyi,wyh,thr)
%FUNCTION
%   z=fp_wft2f(type,f,sigmax,wxl,wxi,wxh,sigmay,wyl,wyi,wyh,thr)
%
%PURPOSE
%   2-D WFT [Fourier version]: Fourier transform is used to compute 
%   convolutions.
%
%INPUT
%   type:   'wff' or 'wfr'
%   f:      2D input signal
%   sigmax: sigma of the window in x, recomended trial value: 10
%   wxl:    low bound of freuqency in x
%   wxi:    increasement of wx
%           recomended trial value: wxi=1/sigmax for wff
%           wxi=0.025 for wfr
%   wxh:    high bound of frequency in x
%   sigmay: sigma of the window in y, recomended trial value: 10
%   wyl:    low bound of freuqency in y
%   wyi:    increasement of wy
%           recomended trial value: wyi=1/sigmay for wff
%           wyi=0.025 for wfr
%   wyh:    high bound of frequency in y
%   thr:    threshold for 'wff', not needed for 'wfr', 
%           recomended trial value:5b for f, 2.5b for fIII and fIV (b is
%           amplitude)
%
%OUTPUT
%   z:      For 'wff'
%           z.filtered is a 2-D filtered signal, 
%           phase=angle(z.filtered) for fI and fIII
%           intensity=real(z.filtered) for fIII and fIV.
%           For 'wfr', 
%           z.wx: local frequency in x
%           z.wy: local frequency in y
%           z.p: phase 
%           z.p_comp: compensated by estimation of c
%           z.b: amplitude
%           z.r: ridge value
%           z.cx: estimation of c in x
%           z.cy: estimation of c in y
%
%EXAMPLES   
%   z = fp_wft2f('wfr',f);
%   z = fp_wft2f('wfr',f,10,-0.5,0.1,0.5,10,-0.5,0.1,0.5);
%   z = fp_wft2f('wff',f);
%   z = fp_wft2f('wff',f,10,-0.5,0.1,0.5,10,-0.5,0.1,0.5,5);
%
%REFERENCE
%[1]Q. Kemao, "Windowed Fourier transform for fringe pattern analysis,"
%   Applied Optics, 43(13):2695-2702, 2004
%[2]Q.Kemao, "Two-dimensional windowed Fourier transform for fringe pattern
%   analysis: principles, applications and implementations,"  
%   Optics and Lasers in Engineering, 45(2): 304-317, 2007.
%[3]Q. Kemao, H. Wang, and W. Gao, Windowed Fourier transform for fringe 
%   pattern analysis: theoretical analyses,Appl. Opt. 47, 5408-5419 (2008).
%
%INFO
%   Last update: 28/07/2011, 05/10/2012, 12/12/2012
%   Contact: mkmqian@ntu.edu.sg (Dr Qian Kemao)
%   Copyright reserved.


%default parameter setting
if nargin==2
    if strcmp(type,'wfr') %wfr algorithm
        sigmax=10; wxl=-2; wxi=0.025; wxh=2;
        sigmay=10; wyl=-2; wyi=0.025; wyh=2;
    elseif strcmp(type,'wff') %wff algorithm
        sigmax=10; wxl=-2-3/sigmax; wxi=1/sigmax; wxh=2+3/sigmax;
        sigmay=10; wyl=-2-3/sigmay; wyi=1/sigmay; wyh=2+3/sigmay;
        thr=6*sqrt(mean2(abs(f).^2)/3);
    end
end

%imaginery unit
jj=sqrt(-1);
%half window size along x, by default 3*sigmax; window size: 2*sx+1
sx=round(3*sigmax); 
%half window size along y, by default 3*sigmay; window size: 2*sy+1
sy=round(3*sigmay); 
%image size
[m n]=size(f);
%expanded size: size(A)+size(B)-1
mm=m+2*sy;nn=n+2*sx;
%meshgrid (2D index) for window
[y x]=meshgrid(-sy:sy,-sx:sx); 
%generate a window g
g=exp(-x.*x/2/sigmax/sigmax-y.*y/2/sigmay/sigmay);
%norm2 normalization
g=g/sqrt(sum(sum(g.*g))); 
%expand f to size [mm nn] and pre-compute its spectrum
Ff=fft2(fexpand(f,mm,nn)); 

%creat a waitbar
h = waitbar(0,'Please wait...');
%total steps for waitbar
steps=floor((wyh-wyl)/wyi)+1;
step=0;
tic;

%wfr    
if strcmp(type,'wfr')
    %to store wx, wy, phase, and ridge value
    z.wx=zeros(m,n); z.wy=zeros(m,n); 
    z.p=zeros(m,n); z.r=zeros(m,n);
    for wyt=wyl:wyi:wyh
        for wxt=wxl:wxi:wxh
            %WFT basis
            gwave=g.*exp(jj*wxt*x+jj*wyt*y);
            %expand w to size [mm nn]
            gwave=fexpand(gwave,mm,nn); 
            %spectrum of w 
            Fg=fft2(gwave); 
            %implement of WFT: conv2(f,w)=ifft2(Ff*Fw);
            sf=ifft2(Ff.*Fg); 
            %cut to get desired data size
            sf=sf(1+sx:m+sx,1+sy:n+sy);
            %indicate where to update
            t=(abs(sf)>z.r); 
            %update r
            z.r=z.r.*(1-t)+abs(sf).*t; 
            %update wx
            z.wx=z.wx.*(1-t)+wxt*t; 
            %update wy
            z.wy=z.wy.*(1-t)+wyt*t; 
            %update phase
            z.p=z.p.*(1-t)+angle(sf).*t; 
        end
            
        %current step for waitbar
        step=step+1;
        %show waitbar
        waitbar(step/steps,h,['WFR2: Elapsed time is ', ...
            num2str(round(toc)),' seconds, please wait...']);
    
        
    end
    
    %Least squares fitting to get cx, with data padding replicating the
    %border pixel
    z.cxx=-conv2(padarray(z.wx,[sx,sy],'replicate'), ...
            x.*g,'same')/sum(sum(x.*x.*g));
    z.cxx=z.cxx(1+sx:m+sx,1+sy:n+sy);
    %Least squares fitting to get cy, with data padding replicating the
    %border pixel
    z.cyy=-conv2(padarray(z.wy,[sx,sy],'replicate'), ...
        y.*g,'same')/sum(sum(y.*y.*g));
    z.cyy=z.cyy(1+sx:m+sx,1+sy:n+sy);
    %phase compensation
    z.p_comp=z.p-0.5*atan(sigmax*sigmax*z.cxx)...
        -0.5*atan(sigmay*sigmay*z.cyy);
    z.p_comp=angle(exp(jj*z.p_comp));

    %scale amplitude
    z.b=z.r.*((1+sigmax^4*z.cxx.^2)/4/pi/sigmax^2).^0.25 ...
        .*((1+sigmay^4*z.cyy.^2)/4/pi/sigmay^2).^0.25;
    
elseif strcmp(type,'wff')
    %to store filtered signal
    z.filtered=zeros(m,n);
    for wyt=wyl:wyi:wyh
        for wxt=wxl:wxi:wxh
%         parfor wxT=0:floor((wxh-wxl)/wxi);
%            wxt=wxT*wxi+wxl;
           %WFT basis
            gwave=g.*exp(jj*wxt*x+jj*wyt*y);
            %expand gwave to size [mm nn]
            gwave=fexpand(gwave,mm,nn); 
            %spectrum of gwave
            Fg=fft2(gwave);  
            %implement of WFT: conv2(f,w)=ifft2(Ff*Fw);
            sf=ifft2(Ff.*Fg); 
            %cut to get desired data size
            sf=sf(1+sx:m+sx,1+sy:n+sy); 
            %threshold the spectrum
            sf=sf.*(abs(sf)>=thr); 
            %exapand sf to size [mm nn]
            sf=fexpand(sf,mm,nn); 
            %implement of IWFT: conv2(sf,gwave);
            filteredt=ifft2(fft2(sf).*Fg); 
            %cut and update
            z.filtered=z.filtered+filteredt(1+sx:m+sx,1+sy:n+sy); 
        end %end wxt

        %current step for waitbar
        step=step+1;
        %show waitbar
        waitbar(step/steps,h,['WFF2: Elapsed time is ', ...
            num2str(round(toc)),' seconds, please wait...']);
            
    end %end wyt
        
    %scale the data
    z.filtered=z.filtered/4/pi/pi*wxi*wyi; 
        
end %end switch type

%store parameter information
z.type=type;
z.f=f;
z.sigmax=sigmax;
z.wxl=wxl;
z.wxi=wxi;
z.wxh=wxh;
z.sigmay=sigmay;
z.wyl=wyl;
z.wyi=wyi;
z.wyh=wyh;
z.sx=sx;
z.sy=sy;
if strcmp(type,'wff')
    z.thr=thr;    
end

%close waitbar
waitbar(1,h,['Elapsed time is ', ...
    num2str(round(toc)),' seconds, please wait...']);
close(h)


function f=fexpand(f,mm,nn)
%expand f to [m n]
%this function can be realized by padarray, but is slower

%size f
[m n]=size(f); 
%store f
f0=f;
%generate a larger matrix with size [mm nn]
f=zeros(mm,nn);
%copy original data
f(1:m,1:n)=f0;
