syms vx vy x y psi bax bay bgz uax uay ugz 'real'

X = [vx;vy;x;y;psi;bax;bay;bgz]
u = [uax;uay;ugz]

f = [cos(psi)*(uax-bax)-sin(psi)*(uay-bay);...
    sin(psi)*(uax-bax)+cos(psi)*(uay-bay);...
    vx;...
    vy;...
    ugz-bgz;...
    0;
    0;
    0;]
  
Fx = jacobian(f,X)

Fu = jacobian(f, u)