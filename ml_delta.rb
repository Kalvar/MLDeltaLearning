# Dynamic loading and matching the PATH of files
$LOAD_PATH.unshift(File.dirname(__FILE__)) unless $LOAD_PATH.include?(File.dirname(__FILE__))  
#Dir["/path/*.rb"].each { |file| require file }

require 'ml_active_function'
require 'ml_active_method'

class MLDelta

  attr_accessor :learning_rate, :max_iteration, :convergence_value, :active_method, :random_scopes, :completion_block

  def initialize
    @active_function   = MLActiveFunction.new
    @active_method     = MLActiveMethod::TANH
    @iteration         = 0
    @sum_error         = 0.0
    @patterns          = []
    @weights           = []
    @targets           = []
    @learning_rate     = 0.5
    @max_iteration     = 1
    @convergence_value = 0.001
    @random_scopes     = [-0.5, 0.5]
  end

  def add_patterns(inputs, target)
    @patterns << inputs
    @targets  << target
  end

  def setup_weights(weights)
    @weights.clear
    @weights += weights
  end

  def setup_random_scopes(min, max)
    @random_scopes.clear
    @random_scopes = [min, max]
  end

  def random_weights
    @weights.clear

    # Follows the inputs count to decide how many weights it needs.
    net_count = @patterns.first.count
    max       = random_scopes.last / net_count
    min       = random_scopes.first / net_count

    net_count.times { @weights << rand(min..max) }
  end

  def training
    @iteration += 1
    @sum_error  = 0.0
    @patterns.each_with_index{ |inputs, index|
      turning_weights_with_inputs(inputs, @targets[index])
    }

    if (@iteration >= @max_iteration) || (calculate_iteration_error <= @convergence_value)
      @completion_block.call( true, @weights, @iteration ) unless @completion_block.nil?
    else
      @iteration_block.call( @iteration, @weights ) unless @iteration_block.nil?
      training
    end
  end

  def training_with_completion(&block)
    @completion_block = block
    training
  end

  def training_with_iteration(iteration_block, completion_block)
    @iteration_block, @completion_block = iteration_block, completion_block
    training
  end

  def direct_output_by_patterns(inputs, &block)
    block.call(net_with_inputs inputs) if block_given?
  end

  private
  def multiply_matrix(matrix, number)
    matrix.map { |obj| obj * number }
  end

  def plus_matrix(matrix, anotherMatrix)
    matrix.collect.with_index { |obj, i| obj + anotherMatrix[i] }
  end

  def activate_output_value(net_output)
    case @active_method
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

  def net_with_inputs(inputs)
    sum = 0.0
    inputs.each.with_index{ |obj, i| sum += ( obj * @weights[i] ) }
    activate_output_value sum
  end

  def dash_of_net(net_output)
    case @active_method
      when MLActiveMethod::SGN
        @active_function.dash_sgn net_output
      when MLActiveMethod::SIGMOID
        @active_function.dash_sigmoid net_output
      when MLActiveMethod::TANH
        @active_function.dash_tanh net_output
      when MLActiveMethod::RBF
        @active_function.dash_rbf net_output, 2.0
      else
        # Nothing else
        net_output
    end
  end

  # Delta defined cost function formula
  def calculate_iteration_error
    (@sum_error / @patterns.count) * 0.5
  end

  def sum_error(error_value)
    @sum_error += (error_value ** 2)
  end

  def turning_weights_with_inputs(inputs, target_value)
    net_output    = net_with_inputs(inputs)
    dash_output   = dash_of_net(net_output)
    error_value   = target_value - net_output

    # new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    simga_value   = @learning_rate * error_value * dash_output
    delta_weights = multiply_matrix(inputs, simga_value)
    new_weights   = plus_matrix(@weights, delta_weights)

    setup_weights(new_weights)
    sum_error(error_value)
  end
end

