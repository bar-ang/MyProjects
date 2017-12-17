function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

I = eye(num_labels);

%feed forward
a1 = [ones(1,size(X',2)); X'];
a2 = sigmoid(Theta1*a1);
a2 = [ones(1,size(a2,2)); a2];
h = sigmoid(Theta2*a2);

%cost function
J = 0;
for i=1:m
  J = J + I(:,y(i))'*log(h(:,i))+(1-I(:,y(i)))'*log(1-h(:,i));
end

J = J * (-1/m);

%regulatization
reg = sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2));
reg = reg * lambda/(2*m);
J = J + reg;

%backpropagation
Del1 = zeros(hidden_layer_size,input_layer_size+1);
Del2 = zeros(num_labels, hidden_layer_size+1);

I = eye(num_labels);
for i=1:m
   a1 = X(i,:)';
   
   %feed forward
    a1 = [ones(1,size(a1,2)); a1];
    a2 = sigmoid(Theta1*a1);
    a2 = [ones(1,size(a2,2)); a2];
    a3 = sigmoid(Theta2*a2);
    
    
    del3 = a3-I(:,y(i));
    del2 = (Theta2'*del3).*(a2.*(1-a2));
    
    Del1 = Del1 + del2(2:end)*a1';
    Del2 = Del2 + del3*a2';
end

Theta1_grad = Del1;
Theta2_grad = Del2;


%gradreg1 = [zeros(1,size(Theta1,2)); Theta1(2:end,:)];
%gradreg2 = [zeros(1,size(Theta2,2)); Theta2(2:end,:)];

%gradreg1 = gradreg1*lambda/m;
%gradreg2 = gradreg2*lambda/m;

gradreg1 = Theta1*lambda;
gradreg2 = Theta2*lambda;

gradreg1(:,1)=zeros(size(gradreg1,1),1);
gradreg2(:,1)=zeros(size(gradreg2,1),1);

Theta1_grad = (Theta1_grad + gradreg1)/m;
Theta2_grad = (Theta2_grad + gradreg2)/m;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end