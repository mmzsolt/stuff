function testnet

x = 0:0.01:6;
x(2,:) = x(1,:)+10;
y = sin(x(1,:)) + sin(x(2,:)).^2;
y(2,:) = x(1,:) + x(2,:);

[w1,b1,w2,b2] = train(x',y');
yt = test(x',w1,b1,w2,b2);
yt = yt';

plot(x(1,:),y(1,:),'-b',x(1,:),y(2,:),'-b',x(1,:),yt(1,:),'-r',x(1,:),yt(2,:),'-r');
%plot(x(1,:),yt(1,:),'-r',x(1,:),yt(2,:),'-r');

end

function [w1,b1,w2,b2] = train(x,y,hn=10)

w1 = randn(size(x,2), hn);
b1 = randn(1, hn);
w2 = randn(hn, size(y,2));
b2 = randn(1, size(y,2));

eta = 0.01;

for epo=1:10
  shufi = randperm(size(x,1));
  for ii=1:length(shufi)
    i = shufi(ii);
    xx=x(i,:)';
    yy=y(i,:)';

    [z1, a1] = forward(w1,b1,xx);
    [z2, a2] = forward(w2,b2,a1);

    db2 = cost_deriv(a2, yy) .* activation_deriv(z2);
    dw2 = a1.*db2';

    db1 = (w2 * db2) .* activation_deriv(z1);
    dw1 = xx.*db1';

    w1 = w1 - eta.*dw1;
    b1 = b1 - eta.*db1';
    w2 = w2 - eta.*dw2;
    b2 = b2 - eta.*db2';
  end
  yy = test(x,w1,b1,w2,b2);
  sum(sum(abs(yy-y)))
end

end

function y = test(x,w1,b1,w2,b2)
y = zeros(size(x,1),size(w2,2));
for i=1:size(x,1)
  [z1, a1] = forward(w1,b1,x(i,:)');
  [x2, a2] = forward(w2,b2,a1);
  y(i,:) = a2;
end
end

function v = activation(v)
v = 1 ./ (1 + exp(v));
end

function v = activation_deriv(v)
v = activation(v).*(1 - activation(v));
end

function v = cost_deriv(x, y)
v = x - y;
end

function [z,a] = forward(w,b,x)
z = x'*w+b;
a = zeros(1,size(w,2));
for i=1:size(w,2)
  a(i) = activation(z(i));
end
z = z';
a = a';
end