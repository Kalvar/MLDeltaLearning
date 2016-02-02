require './ml_active_function'
require './ml_active_method'

DEFAULT_RANDOM_MAX = 0.5
DEFAULT_RANDOM_MIN = -0.5

class MLDelta
  @@sharedDelta = MLDelta.new
  attr_accessor :learning_rate, :max_iteration, :convergence_value
  attr_accessor :activeMethod

  attr_accessor :iterationBlock
  attr_accessor :completionBlock

  def initialize
    @active_function = MLActiveFunction.new
    @iteration       = 0
    @sum_error        = 0.0
    @patterns         = []
    @weights          = []
    @targets          = []
    @learning_rate     = 0.5
    @max_iteration     = 1
    @convergence_value = 0.001
    @activeMethod     = MLActiveMethod::TANH

    @iterationBlock   = nil
    @completionBlock  = nil
  end

  # Public methods
  public

  def self.sharedDelta
    return @@sharedDelta
  end

  def add_patterns(inputs, target)
    @patterns << inputs
    @targets << target
  end

  def setup_weights(weights)
    @weights.clear if @weights.count > 0
    @weights += weights
  end

  def randomWeights
    @weights.clear if @weights.count > 0

    # Follows the inputs count to decide how many weights it needs.
    randomMaker = Random.new
    input_net_count = @patterns.first().count
    input_max = DEFAULT_RANDOM_MAX / input_net_count
    input_min = DEFAULT_RANDOM_MIN / input_net_count
    for i in (0..._inputNetCount)
      @weights.push(randomMaker.rand(input_min..input_max))
    end
  end

  def training
    @iteration += 1
    @sum_error = 0.0
    @patterns.each_with_index{ |inputs, patternIndex| _turningWeightsWithInputs(inputs, @targets[patternIndex]) }

    if (@iteration >= @max_iteration) || (_calculateIterationError() <= @convergence_value)
      @completionBlock.call( true, @weights, @iteration ) unless @completionBlock.nil?
    else
      @iterationBlock.call( @iteration, @weights ) unless @iterationBlock.nil?
      training
    end
  end

  def training_with_completion(&block)
    @completionBlock = block
    training
  end

  def training_with_iteration(iterationBlock, completionBlock)
    @iterationBlock = iterationBlock
    @completionBlock = completionBlock
    training
  end

  def direct_output_by_patterns(inputs, &block)
    block.call(_fOfNetWithInputs(inputs)) if block_given?
  end

  private
  def multiply_matrix(matrix, number)
    matrix.map{ |obj| obj * number }
  end

  def plus_matrix(matrix, anotherMatrix)
    matrix.collect.with_index{ |obj, i| obj + anotherMatrix[i] }
  end

  def _activateOutputValue(net_output)
    case @activeMethod
      when MLActiveMethod::SGN
        @active_function.sgn net_output
      when MLActiveMethod::SIGMOID
        @active_function.sigmoid net_output
      when MLActiveMethod::TANH
        @active_function.tanh net_output
      when MLActiveMethod::RBF
        @active_function.rbf net_output, 2.0
      else
        # Nothing else
        net_output
    end
  end

  def _fOfNetWithInputs(inputs)
    # to_f
    sum = 0.0
    inputs.each.with_index{ |obj, i| sum += ( obj * @weights[i] ) }
    _activateOutputValue(sum)
  end

  def _fDashOfNet(net_output)
    case @activeMethod
      when MLActiveMethod::SGN
        @active_function.dashSgn(net_output)
      when MLActiveMethod::SIGMOID
        @active_function.dashSigmoid(net_output)
      when MLActiveMethod::TANH
        @active_function.dashTanh(net_output)
      when MLActiveMethod::RBF
        #@active_function.dashRbf(net_output)
      else
        # Nothing else
        net_output
    end
  end

  # Delta defined cost function formula
  def _calculateIterationError
    (@sum_error / @patterns.count) * 0.5
  end

  def sum_error(error_value)
    @sum_error += (error_value ** 2)
  end

  def _turningWeightsWithInputs(inputs, target_value)
    net_output = _fOfNetWithInputs(inputs)
    dash_output = _fDashOfNet(net_output)
    error_value = target_value - net_output

    # new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    simga_value = @learning_rate * error_value * dash_output
    delta_weights = multiply_matrix(inputs, simga_value)
    new_weights = plus_matrix(@weights, delta_weights)

    setup_weights(new_weights)
    sum_error(error_value)
  end
end

# Use methods
delta                  = MLDelta.new
delta.activeMethod     = MLActiveMethod::TANH
delta.learning_rate     = 0.8
delta.convergence_value = 0.001
delta.max_iteration     = 1000
delta.add_patterns([1.0, -2.0, 0.0, -1.0], -1.0)
delta.add_patterns([0.0, 1.5, -0.5, -1.0], 1.0)
delta.setup_weights([1.0, -1.0, 0.0, 0.5])
#delta.randomWeights()

iterationBlock = Proc.new do |iteration, weights|
  puts "iteration : #{iteration}, weights : #{weights}"
end

completionBlock = Proc.new do |success, weights, totalIteration|
  puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
  delta.direct_output_by_patterns([1.0, -2.0, 0.0, -1.0]){ |predication| puts "predication result is #{predication}" }
end

delta.training_with_iteration(iterationBlock, completionBlock)

# delta.training_with_completion {
#   |success, weights, totalIteration|
#   puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
# }
