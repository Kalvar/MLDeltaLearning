require './ml_delta'

delta = MLDelta.new
delta.activeMethod = MLActiveMethod::TANH
delta.learning_rate = 0.8
delta.convergence_value = 0.001
delta.max_iteration = 1000
delta.add_patterns [1.0, -2.0, 0.0, -1.0], -1.0
delta.add_patterns [0.0, 1.5, -0.5, -1.0], 1.0
delta.setup_weights [1.0, -1.0, 0.0, 0.5]
#delta.random_weights()

iteration_block = Proc.new do |iteration, weights|
  puts "iteration : #{iteration}, weights : #{weights}"
end

completion_block = Proc.new do |success, weights, totalIteration|
  puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
  delta.directOutputByPatterns([1.0, -2.0, 0.0, -1.0]){ |predication| puts "predication result is #{predication}" }
end

delta.training_with_iteration(iteration_block, completion_block)

# delta.trainingWithCompletion {
#   |success, weights, totalIteration|
#   puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
# }
