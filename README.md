## About

MLDelta is implemented by Ruby and Delta Learning Method in Machine Learning that is also a supervisor and used gradient method to find out the best solution.

## How To Get Started

#### Require
``` ruby
require './ml_delta'
```

#### Normal Case
``` ruby
delta                  = MLDelta.new
delta.activeMethod     = MLActiveMethod::TANH
delta.learningRate     = 0.8
delta.convergenceValue = 0.001
delta.maxIteration     = 1000
delta.addPatterns([1.0, -2.0, 0.0, -1.0], -1.0)
delta.addPatterns([0.0, 1.5, -0.5, -1.0], 1.0)
delta.setupWeights([1.0, -1.0, 0.0, 0.5])

# Setting the block of per iteration training
iterationBlock = Proc.new do |iteration, weights|
  puts "iteration : #{iteration}, weights : #{weights}"
end

# Setting the block of completion when it finish training
completionBlock = Proc.new do |success, weights, totalIteration|
  puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
  # Verifying the pattern
  delta.directOutputByPatterns([1.0, -2.0, 0.0, -1.0]){ |predication| puts "predication result is #{predication}" }
end

# Start in training
delta.trainingWithIteration(iterationBlock, completionBlock)
```

#### Lazy Case
``` ruby
# 1. If you wish automatic setups all weights of pattern that you could use delta.randomWeights() to instead of delta.setupWeights().
# 2. If you just wanna see the result without iteration running that you could directly use the method as below :
	delta.trainingWithCompletion { 
	   |success, weights, totalIteration| 
	   puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
	}
```

## Version

V1.0

## LICENSE

MIT.

