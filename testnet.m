function testnet

x = 0:0.01:6;
x(2,:) = x(1,:);
y = sin(x(1,:)) + sin(x(2,:)).^2;

train(x',y');

%plot(x(1,:),y,'-b');

end

function [w1,w2] = train(x,y,hn=10)

w1 = randn(size(x,2), hn);
w2 = randn(hn, size(y,2));
hl = zeros(hn,1);

xx=x(5,:)';

activation(w1,xx,1)

end

function v = activation(w,x,i)
v = w(:,i)'*x;
v = 1 / (1 + exp(v));
end

