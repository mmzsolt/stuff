function testnet

x = 0:0.01:6;
x(2,:) = x(1,:)+10;
y = sin(x(1,:)) + sin(x(2,:)).^2;

train(x',y');

%plot(x(1,:),y,'-b');

end

function [w1,w2] = train(x,y,hn=10)

w1 = randn(size(x,2), hn);
w2 = randn(hn, size(y,2));

xx=x(5,:)';

hl = forward(w1,xx)
yy = forward(w2,hl)

end

function v = activation(w,x,i)
v = w(:,i)'*x;
v = 1 / (1 + exp(v));
end

function o = forward(w,x)
o = zeros(size(w,2),1);
for i=1:size(w,2)
  o(i) = activation(w,x,i);
end
end