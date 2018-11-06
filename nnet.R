sigmoid <- function(z) 1/(1+exp(-z))
sigmoid_prime <- function(z){
  sz = sigmoid(z)
  sz*(1-sz)
}

logistic_reg <- function(X, y, epochs, lr){
  activation = as.matrix(X)
  p = ncol(activation)
  n = nrow(activation)
  
  w_hat = rnorm(p) # initializing at zero can cause problems
  b_hat = rnorm(1)
  for (j in 1:epochs) {
    delta = sigmoid(activation %*% w_hat + b_hat) - y
    # Update weights with gradient descent
    nabla_w = crossprod(activation, delta) / n
    nabla_b = mean(delta)
    w_hat = w_hat - lr*nabla_w
    b_hat <- b_hat - (lr*nabla_b)
  }
  c(b_hat, w_hat)
}

neural_net <- function(X, y, epochs, lr, 
                       architecture = c(10,5,3)){
  # X is (p x n)
  # y is dummy matrix, (nclass x n)
  # architecture is c(nvars,...,nclasses)
  n = ncol(y)
  num_layers = length(architecture)
  listw = architecture[-num_layers] # Skip last (weights from 1st to 2nd-to-last)
  listb = architecture[-1]  # Skip first element (biases from 2nd to last)
  
  # Initialise with gaussian distribution for biases and weights
  biases <- lapply(seq_along(listb), 
                   function(idx) rnorm(listb[[idx]])
  )
  
  weights <- lapply(seq_along(listw), function(idx){
    c <- listw[[idx]]
    r <- listb[[idx]]
    matrix(rnorm(n=r*c), nrow=r, ncol=c)
  })
  
  # Using GD rather than SGD for simplicity
  for(j in 1:epochs){
    # Initialise updates with zero vectors
    nabla_b <- vector('list', length = length(listb))
    nabla_w <- vector('list', length = length(listw))
    
    # Step 1: Feed-forward (get predictions)
    activations <- list()
    activations[[1]] <- activation <- X
    # z = sigmoid(w * x + b)
    # So need zs to store all z-vectors
    zs <- list()
    for(f in 1:length(biases)){
      b = biases[[f]]
      w = weights[[f]]
      zs[[f]] <- z <- w %*% activation  + b
      activations[[f+1]] <- activation <- sigmoid(z)
      # Activations already contain one element
    }
    # Step 2: Backwards (update gradient using errors)
    # Last layer
    delta = (activation - y) 
    nabla_b[[length(nabla_b)]] = rowSums(delta)
    nabla_w[[length(nabla_w)]] = tcrossprod(delta, activations[[length(activations)-1]])
    # Second to second-to-last-layer
    # If no hidden-layer reduces to multinomial logit
    if (num_layers > 2) {
      for (k in 2:(num_layers-1)) {
        sp = sigmoid_prime(zs[[length(zs)-(k-1)]])
        delta = crossprod(weights[[length(weights)-(k-2)]], delta) * sp
        nabla_b[[length(nabla_b)-(k-1)]] = rowSums(delta)
        testyy = activations[[length(activations)-k]]
        nabla_w[[length(nabla_w)-(k-1)]] = tcrossprod(delta, testyy)
      }
    }
    weights <- mapply(function(w,nw) w - (lr/n)*nw, weights, nabla_w)
    biases <- mapply(function(b,nb) b - (lr/n)*nb, biases, nabla_b)
  }
  # Final feed forward for predictions
  activation = X
  for(f in 1:length(biases)){
    b = biases[[f]]
    w = weights[[f]]
    z <- w %*% activation  + b
    activation <- sigmoid(z)
  }
  return(list(weights=weights, biases=biases, activation=activation))
}
