function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

a1 = [ones(m, 1) X]; % adding required bias to activation
a2 = [ones(m, 1) sigmoid(Theta1 * a1')']; % computing Hidden Layer
a3 = sigmoid(Theta2 * a2')'; %Computing Output Layer
[v p] = max(a3, [], 2);


end
