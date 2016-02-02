## About

MLDelta is implemented by Ruby and Delta Learning Method in Machine Learning that is also a supervisor and used gradient method to find out the best solution.

## How To Get Started

#### Require
``` ruby
require './ml_delta'
```

#### Normal Case
``` ruby
delta = MLDelta.new
delta.active_method = MLActiveMethod::TANH
delta.learning_rate = 0.8
delta.convergence_value = 0.001
delta.max_iteration = 1000
delta.add_patterns([1.0, -2.0, 0.0, -1.0], -1.0)
delta.add_patterns([0.0, 1.5, -0.5, -1.0], 1.0)
delta.setup_weights([1.0, -1.0, 0.0, 0.5])

# Setting the block of per iteration training
iteration_block = Proc.new do |iteration, weights|
  puts "iteration : #{iteration}, weights : #{weights}"
end

# Setting the block of completion when it finish training
completion_block = Proc.new do |success, weights, total_iteration|
  puts "success : #{success}, weights : #{weights}, total_iteration : #{total_iteration}"

  # Verifying the pattern
  delta.direct_output_by_patterns [1.0, -2.0, 0.0, -1.0] { |predication|
    puts "predication result is #{predication}"
  }
end

# Start in training
delta.training_with_iteration(iteration_block, completion_block)
```

#### Lazy Case
1. If you wish automatic setups all weights of pattern that you could use delta.random_weights() to instead of delta.setup_weights().
2. If you just wanna see the result without iteration running that you could directly use the method as below :

``` ruby
delta.training_with_completion {
  |success, weights, total_iteration|
  puts "success : #{success}, weights : #{weights}, total_iteration : #{total_iteration}"
}
```

## Version

V1.0

## LICENSE

MIT.

