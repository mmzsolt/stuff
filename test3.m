function test3

n = 250;

z = linspace(0,4*pi,n)';
x = 2*cos(z) + rand(1,n)';
y = 2*sin(z) + rand(1,n)';

%a = 0.11;
%b = 0.22;
%c = 0.33;
%d = 1;
%x = randn(n, 1);
%y = randn(n, 1);
%z = pln(x, y) + (randn(n,1) * 0.1);

scatter3(x,y,z)

co = est(x, y ,z)
zz = use(x, y, z, co);

hold on
scatter3(x, y, zz, [], 'r')

function co = est(x, y, z)
  xx = phi(x, y, z);
  aa = xx'*xx;
  bb = xx'*z;
  co = linsolve(aa,bb);
  %co = (z\xx)';
end

function zz = use(x, y, z, co)
  xx = phi(x, y, z);
  zz = xx * co;
end

function xx = phi(x, y, z)
  xx = [x, y, ones(size(x,1),1)];
end

function z = pln(x,y)
  z = (a.*x + b.*y + d) / c;
end

end

